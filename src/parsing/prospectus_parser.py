from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf

from src.config import get_settings
from src.parsing.base import BaseDocumentParser
from src.parsing.chunking import (
    BBox,
    RawChunk,
    TextChunker,
    join_lines,
    normalize_rows,
    overlap_ratio,
    rows_to_markdown_chunks,
    table_quality,
)
from src.parsing.schemas import ChunkType, DocumentChunk
from src.utils import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class _Block:
    text: str
    bbox: BBox
    size: float
    bold: bool
    page: int


@dataclass
class _Table:
    index: int
    bbox: BBox
    rows: list[list[str]]


class FullProspectusParser(BaseDocumentParser):
    """Parseur universel pour prospectus longs.

    Par rapport à la version d'origine :
    - les tableaux sont retirés du flux de texte (plus de doublon TABLE + TEXT) ;
    - les tableaux sont convertis en Markdown (l'image ne part au VLM que si le Markdown est douteux) ;
    - le texte est découpé en flux continu sur tout le document (un chunk peut chevaucher deux pages) ;
    - les titres sont détectés par la taille/graisse de police et préfixent chaque chunk (« section ») ;
    - les en-têtes / pieds de page répétés sont supprimés ;
    - la légende située juste au-dessus d'un tableau est rattachée à ce tableau.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        table_chunk_size: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
        table_dpi: Optional[int] = None,
    ):
        super().__init__(output_dir)
        s = get_settings()
        self.chunk_size = chunk_size or s.chunk_size
        self.chunk_overlap = s.chunk_overlap if chunk_overlap is None else chunk_overlap
        self.table_chunk_size = table_chunk_size or s.table_chunk_size
        self.min_chunk_chars = min_chunk_chars or s.min_chunk_chars
        self.table_dpi = table_dpi or s.table_dpi

    # ------------------------------------------------------------------ API
    def extract_chunks(self, pdf_path: str, doc_id: Optional[str] = None) -> list[DocumentChunk]:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")
        doc_id = self._resolve_doc_id(pdf_file, doc_id)

        doc = pymupdf.open(pdf_file)
        try:
            raw_chunks = self._parse(doc, doc_id)
            page_sizes = [(p.rect.width, p.rect.height) for p in doc]
        finally:
            doc.close()

        chunks: list[DocumentChunk] = []
        for raw in raw_chunks:
            if raw.kind == "text" and len(raw.text) < 30:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id="",
                    doc_id=doc_id,
                    source_document=pdf_file.name,
                    page_number=raw.page_start,
                    page_end=raw.page_end if raw.page_end != raw.page_start else None,
                    chunk_type=ChunkType.TABLE if raw.kind == "table" else ChunkType.TEXT,
                    section=raw.section,
                    text_content=raw.text,
                    image_path=raw.image_path,
                    needs_image=raw.needs_image,
                    bbox=raw.bbox or (0.0, 0.0, *page_sizes[raw.page_start - 1]),
                )
            )
        return self._finalize(chunks, doc_id)

    # ------------------------------------------------------------- parsing
    def _parse(self, doc: pymupdf.Document, doc_id: str) -> list[RawChunk]:
        pages_blocks = [self._read_blocks(doc[i], i + 1) for i in range(len(doc))]
        heights = [doc[i].rect.height for i in range(len(doc))]
        body_size = self._body_font_size(pages_blocks)
        noise = self._detect_noise(pages_blocks, heights)

        chunker = TextChunker(self.chunk_size, self.chunk_overlap, self.min_chunk_chars)
        headings: list[tuple[int, str]] = []

        for page_index in range(len(doc)):
            page, number = doc[page_index], page_index + 1
            blocks = [
                b for b in pages_blocks[page_index] if not self._is_noise(b, heights[page_index], noise)
            ]
            tables = self._extract_tables(page, number)
            if tables:  # on retire du flux de texte ce qui appartient déjà à un tableau
                blocks = [b for b in blocks if not any(overlap_ratio(b.bbox, t.bbox) > 0.5 for t in tables)]

            items: list[tuple[float, int, str, object]] = [(b.bbox[1], 0, "text", b) for b in blocks]
            items += [(t.bbox[1], 1, "table", t) for t in tables]
            items.sort(key=lambda it: (it[0], it[1]))

            for _, _, kind, obj in items:
                if kind == "table":
                    self._emit_table(chunker, page, obj, number, doc_id, self._section(headings))  # type: ignore[arg-type]
                    continue
                block: _Block = obj  # type: ignore[assignment]
                level = self._heading_level(block, body_size)
                if level is not None:
                    self._push_heading(headings, level, block.text)
                    chunker.add(block.text, number, block.bbox, self._section(headings), is_heading=True)
                else:
                    chunker.add(block.text, number, block.bbox, self._section(headings))

        chunker.flush()
        return chunker.chunks

    # -------------------------------------------------------- lecture blocs
    def _read_blocks(self, page: pymupdf.Page, number: int) -> list[_Block]:
        data = page.get_text("dict", sort=True)
        blocks: list[_Block] = []
        for raw in data.get("blocks", []):
            if raw.get("type") != 0:
                continue
            lines: list[str] = []
            weights: list[tuple[float, int, bool]] = []
            for line in raw.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue
                lines.append(text)
                for span in line["spans"]:
                    n = len(span["text"].strip())
                    if n:
                        is_bold = bool(span["flags"] & 16) or "bold" in span.get("font", "").lower()
                        weights.append((span["size"], n, is_bold))
            total = sum(n for _, n, _ in weights)
            if not lines or total == 0:
                continue
            size = sum(s * n for s, n, _ in weights) / total
            bold = sum(n for _, n, b in weights if b) / total > 0.6
            x0, y0, x1, y1 = raw["bbox"]
            blocks.append(_Block(join_lines(lines), (x0, y0, x1, y1), size, bold, number))
        return blocks

    @staticmethod
    def _body_font_size(pages_blocks: list[list[_Block]]) -> float:
        counter: Counter[float] = Counter()
        for blocks in pages_blocks:
            for b in blocks:
                counter[round(b.size * 2) / 2] += len(b.text)
        return counter.most_common(1)[0][0] if counter else 10.0

    # ------------------------------------------- en-têtes / pieds de page
    @staticmethod
    def _signature(text: str) -> str:
        return re.sub(r"\d+", "#", normalize_text(text))

    def _detect_noise(self, pages_blocks: list[list[_Block]], heights: list[float]) -> set[str]:
        counter: Counter[str] = Counter()
        for blocks, height in zip(pages_blocks, heights):
            seen = {self._signature(b.text) for b in blocks if self._in_margin_band(b, height)}
            counter.update(seen)
        threshold = max(3, int(0.3 * len(pages_blocks)))
        return {sig for sig, count in counter.items() if count >= threshold}

    @staticmethod
    def _in_margin_band(block: _Block, page_height: float) -> bool:
        return block.bbox[3] < page_height * 0.09 or block.bbox[1] > page_height * 0.91

    def _is_noise(self, block: _Block, page_height: float, noise: set[str]) -> bool:
        return self._in_margin_band(block, page_height) and self._signature(block.text) in noise

    # ------------------------------------------------------------- titres
    @staticmethod
    def _looks_like_heading(block: _Block, body_size: float) -> bool:
        text = block.text.strip()
        if len(text) < 3 or len(text) > 120 or text.count("\n") > 1 or text.isdigit():
            return False
        if text.endswith((".", ",", ";")):
            return False
        bigger = block.size >= body_size * 1.12
        bold_line = block.bold and block.size >= body_size * 0.95 and len(text) <= 100
        caps = text.isupper() and len(text) >= 4 and block.size >= body_size * 0.95
        return bigger or bold_line or caps

    def _heading_level(self, block: _Block, body_size: float) -> Optional[int]:
        if not self._looks_like_heading(block, body_size):
            return None
        ratio = block.size / body_size if body_size else 1.0
        if ratio >= 1.5:
            return 0
        if ratio >= 1.25:
            return 1
        if ratio >= 1.12:
            return 2
        return 3

    @staticmethod
    def _push_heading(stack: list[tuple[int, str]], level: int, text: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()[:120]
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))

    @staticmethod
    def _section(stack: list[tuple[int, str]]) -> Optional[str]:
        if not stack:
            return None
        return " > ".join(t for _, t in stack[-3:])[:200]

    # ----------------------------------------------------------- tableaux
    def _extract_tables(self, page: pymupdf.Page, number: int) -> list[_Table]:
        try:
            found = list(page.find_tables().tables)
        except Exception as exc:  # find_tables peut échouer sur des pages atypiques
            logger.warning("find_tables a échoué (page %s) : %s", number, exc)
            return []
        tables: list[_Table] = []
        for idx, tab in enumerate(found):
            try:
                rows = normalize_rows(tab.extract())
            except Exception:
                continue
            if len(rows) < 2 or len(rows[0]) < 2:
                continue  # une « grille » à une ligne/colonne est du texte mis en forme, pas un tableau
            if sum(len(c) for r in rows for c in r) < 25:
                continue
            b = tab.bbox
            tables.append(_Table(idx, (b[0], b[1], b[2], b[3]), rows))
        return tables

    def _emit_table(
        self,
        chunker: TextChunker,
        page: pymupdf.Page,
        table: _Table,
        number: int,
        doc_id: str,
        section: Optional[str],
    ) -> None:
        image_path: Optional[str] = None
        try:
            path = self.output_dir / f"{doc_id}_p{number}_tab{table.index}.png"
            pix = page.get_pixmap(clip=pymupdf.Rect(*table.bbox), dpi=self.table_dpi)
            pix.save(str(path))
            image_path = str(path)
        except Exception as exc:
            logger.warning("Capture du tableau impossible (page %s) : %s", number, exc)

        caption = chunker.pop_caption(number, table.bbox[1])
        chunker.flush()
        reliable = table_quality(table.rows)
        for markdown in rows_to_markdown_chunks(table.rows, self.table_chunk_size):
            text = f"{caption}\n{markdown}" if caption else markdown
            chunker.emit_external(
                RawChunk(
                    kind="table",
                    text=text,
                    page_start=number,
                    page_end=number,
                    bbox=table.bbox,
                    section=section,
                    image_path=image_path,
                    needs_image=(not reliable) and image_path is not None,
                )
            )
