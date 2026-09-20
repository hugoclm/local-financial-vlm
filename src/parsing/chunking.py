"""Logique de découpage pure (sans PyMuPDF ni Pydantic), donc testable isolément.

- `TextChunker` regroupe des paragraphes en chunks de taille bornée, avec recouvrement,
  en continu sur tout le document (un chunk peut chevaucher deux pages) et en respectant
  les frontières de section.
- `rows_to_markdown_chunks` transforme un tableau en Markdown (bien plus lisible pour un LLM
  et pour l'embedding que le texte aplati cellule par cellule), en répétant l'en-tête si
  le tableau doit être scindé.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

BBox = tuple[float, float, float, float]


# --------------------------------------------------------------------------------------
# Géométrie / lignes
# --------------------------------------------------------------------------------------
def overlap_ratio(a: BBox, b: BBox) -> float:
    """Part de l'aire de `a` recouverte par `b` (0..1)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    if area_a <= 0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / area_a


_BULLET = re.compile(r"^(?:[•●▪■◦·*\-–—]|\(?\d{1,2}[.)]|[a-zA-Z][.)])\s+")


def join_lines(lines: list[str]) -> str:
    """Recolle les lignes d'un même bloc PDF : dé-césure, puces conservées sur leur ligne."""
    if not lines:
        return ""
    out = lines[0]
    for prev, line in zip(lines, lines[1:]):
        if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha() and line[:1].islower():
            out = out[:-1] + line  # « inves- » + « tissement »
        elif _BULLET.match(line):
            out += "\n" + line
        else:
            out += " " + line
    return out


# --------------------------------------------------------------------------------------
# Découpage de texte
# --------------------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+")


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut < max_chars * 0.5:
            cut = max_chars
        out.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        out.append(text)
    return out


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Découpe un texte trop long sur les fins de phrase (puis sur les mots en dernier recours).

    Corrige le comportement d'origine où un paragraphe > chunk_size n'était tronqué qu'une fois.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    units: list[tuple[str, str]] = []  # (texte, séparateur avant)
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            units.append((line, "\n"))
            continue
        first = True
        for sentence in _SENT_SPLIT.split(line):
            for piece in _hard_wrap(sentence, max_chars):
                units.append((piece, "\n" if first else " "))
                first = False

    parts: list[str] = []
    current = ""
    for unit, sep in units:
        if not current:
            current = unit
        elif len(current) + len(sep) + len(unit) <= max_chars:
            current += sep + unit
        else:
            parts.append(current)
            current = unit
    if current:
        parts.append(current)
    return parts


def tail_at_boundary(text: str, n: int) -> str:
    """Les ~n derniers caractères, alignés sur un début de phrase ou de mot (recouvrement propre)."""
    if n <= 0 or len(text) <= n:
        return ""
    tail = text[-n:]
    match = re.search(r"[.!?;:]\s+|\s+", tail)
    if match and match.end() < len(tail) - 20:
        tail = tail[match.end():]
    return tail.strip()


@dataclass
class RawChunk:
    kind: str  # "text" | "table"
    text: str
    page_start: int
    page_end: int
    bbox: BBox
    section: Optional[str] = None
    image_path: Optional[str] = None
    needs_image: bool = False


@dataclass
class _Piece:
    text: str
    page: int
    bbox: BBox
    is_overlap: bool = False


class TextChunker:
    """Assemble des paragraphes en chunks ≤ ~chunk_size (+ recouvrement) sur tout le document.

    Un chunk peut atteindre `chunk_size + overlap` caractères (le recouvrement s'ajoute en tête).
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 150, min_chunk_chars: int = 250) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_chars = min_chunk_chars
        self.chunks: list[RawChunk] = []
        self._pieces: list[_Piece] = []
        self._section: Optional[str] = None

    # -- longueurs -------------------------------------------------------------------
    def _length(self, pieces: list[_Piece]) -> int:
        return sum(len(p.text) for p in pieces) + 2 * max(0, len(pieces) - 1)

    def _real(self) -> list[_Piece]:
        return [p for p in self._pieces if not p.is_overlap]

    # -- API -------------------------------------------------------------------------
    def add(self, text: str, page: int, bbox: BBox, section: Optional[str], is_heading: bool = False) -> None:
        text = (text or "").strip()
        if not text:
            return
        # Un titre ouvre un nouveau chunk, sauf si le chunk courant est trop petit pour tenir seul.
        if is_heading and self._length(self._real()) >= self.min_chunk_chars:
            self.flush()
        for part in split_long_text(text, self.chunk_size):
            self._add_piece(part, page, bbox, section)

    def flush(self) -> None:
        """Émet le chunk en cours (sans recouvrement avec le suivant)."""
        self._emit(carry_overlap=False)

    def emit_external(self, raw: RawChunk) -> None:
        """Insère un chunk déjà construit (ex. tableau) en préservant l'ordre du document."""
        self.flush()
        self.chunks.append(raw)

    def pop_caption(
        self, page: int, table_top: float, max_gap: float = 45.0, max_chars: int = 250
    ) -> Optional[str]:
        """Retire du tampon le court paragraphe juste au-dessus d'un tableau (sa légende)."""
        if not self._pieces:
            return None
        last = self._pieces[-1]
        if last.is_overlap or last.page != page or len(last.text) > max_chars:
            return None
        if not (table_top - max_gap <= last.bbox[3] <= table_top + 5):
            return None
        self._pieces.pop()
        if not self._real():
            self._pieces = []
        return last.text

    # -- interne ---------------------------------------------------------------------
    def _add_piece(self, text: str, page: int, bbox: BBox, section: Optional[str]) -> None:
        if self._pieces and self._length(self._pieces) + 2 + len(text) > self.chunk_size:
            self._emit(carry_overlap=True)
        if not self._real():
            self._section = section
        self._pieces.append(_Piece(text, page, bbox))

    def _emit(self, carry_overlap: bool) -> None:
        real = self._real()
        if not real:
            self._pieces = []
            return
        text = "\n\n".join(p.text for p in self._pieces)
        first, last = real[0], real[-1]
        same_page = [p.bbox for p in real if p.page == first.page]
        bbox = (
            min(b[0] for b in same_page),
            min(b[1] for b in same_page),
            max(b[2] for b in same_page),
            max(b[3] for b in same_page),
        )
        self.chunks.append(
            RawChunk(
                kind="text",
                text=text,
                page_start=first.page,
                page_end=last.page,
                bbox=bbox,
                section=self._section,
            )
        )
        tail = tail_at_boundary(text, self.overlap) if carry_overlap else ""
        self._pieces = [_Piece(tail, last.page, last.bbox, is_overlap=True)] if tail else []


# --------------------------------------------------------------------------------------
# Tableaux
# --------------------------------------------------------------------------------------
def _clean_cell(cell: object) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell).replace("|", "/")).strip()


def normalize_rows(rows: list[list[object]]) -> list[list[str]]:
    """Nettoie un tableau brut : cellules en texte, lignes et colonnes entièrement vides retirées."""
    cleaned = [[_clean_cell(c) for c in row] for row in (rows or [])]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return []
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    keep = [j for j in range(width) if any(r[j] for r in cleaned)]
    return [[r[j] for j in keep] for r in cleaned]


def table_quality(rows: list[list[str]]) -> bool:
    """Vrai si le Markdown extrait est probablement fiable (sinon on enverra l'image au VLM)."""
    if len(rows) < 2 or len(rows[0]) < 2:
        return False
    cells = [c for r in rows for c in r]
    empty_ratio = sum(1 for c in cells if not c) / len(cells)
    return empty_ratio <= 0.45


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def rows_to_markdown_chunks(rows: list[list[str]], max_chars: int) -> list[str]:
    """Tableau → un ou plusieurs blocs Markdown (l'en-tête est répété si on doit scinder)."""
    if not rows:
        return []
    header = rows[0]
    head_md = _md_row(header) + "\n" + _md_row(["---"] * len(header))
    body = rows[1:]
    if not body:
        return [head_md]

    parts: list[str] = []
    lines: list[str] = []
    size = len(head_md)
    for row in body:
        line = _md_row(row)
        if lines and size + len(line) + 1 > max_chars:
            parts.append(head_md + "\n" + "\n".join(lines))
            lines, size = [], len(head_md)
        lines.append(line)
        size += len(line) + 1
    if lines:
        parts.append(head_md + "\n" + "\n".join(lines))
    return parts
