from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pymupdf

from src.config import get_settings
from src.parsing.base import BaseDocumentParser
from src.parsing.chunking import (
    RawChunk,
    TextChunker,
    normalize_rows,
    overlap_ratio,
    rows_to_markdown_chunks,
)
from src.parsing.schemas import ChunkType, DocumentChunk

logger = logging.getLogger(__name__)


class KIDParser(BaseDocumentParser):
    """Parseur haute fidélité pour DIC / KID PRIIPs (3-4 pages).

    Les 4 sections cibles donnent chacune un chunk TABLE (texte + tableaux en Markdown + capture
    haute résolution). Le reste du texte est regroupé en chunks de taille normale au lieu d'un
    mini-chunk par bloc.
    """

    TARGET_SECTIONS = [
        "INDICATEUR DE RISQUE",
        "SCÉNARIOS DE PERFORMANCE",
        "COÛTS AU FIL DU TEMPS",
        "COMPOSITION DES COÛTS",
    ]

    def __init__(self, output_dir: Optional[str] = None, dpi: Optional[int] = None):
        super().__init__(output_dir)
        self.dpi = dpi or get_settings().table_dpi

    # ------------------------------------------------------------------ API
    def extract_chunks(self, pdf_path: str, doc_id: Optional[str] = None) -> list[DocumentChunk]:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")
        doc_id = self._resolve_doc_id(pdf_file, doc_id)
        settings = get_settings()

        doc = pymupdf.open(pdf_file)
        raw_chunks: list[RawChunk] = []
        try:
            for page_index in range(len(doc)):
                page, number = doc[page_index], page_index + 1
                raw_chunks.extend(self._parse_page(page, number, pdf_file, doc_id, settings))
        finally:
            doc.close()

        chunks = [
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
                bbox=raw.bbox,
            )
            for raw in raw_chunks
        ]
        return self._finalize(chunks, doc_id)

    # -------------------------------------------------------------- interne
    def _match_section(self, text: str) -> Optional[str]:
        upper = text.upper()
        return next((sec for sec in self.TARGET_SECTIONS if sec in upper), None)

    def _parse_page(self, page, number: int, pdf_file: Path, doc_id: str, settings) -> list[RawChunk]:
        blocks = page.get_text("blocks", sort=True)  # tri vertical puis horizontal
        chunker = TextChunker(settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars)
        consumed: set[int] = set()

        for i, block in enumerate(blocks):
            if i in consumed or block[6] != 0:
                continue
            text = block[4].strip()
            bbox = (block[0], block[1], block[2], block[3])
            if not text or (text.startswith("Page ") and len(text) < 15):
                continue

            matched = self._match_section(text)
            if not matched:
                chunker.add(text, number, bbox, None)
                continue

            # Section cible : on absorbe les blocs suivants jusqu'au prochain titre.
            section_blocks = [block]
            consumed.add(i)
            for j in range(i + 1, len(blocks)):
                nxt = blocks[j]
                nxt_text = nxt[4].strip()
                is_next_title = self._match_section(nxt_text) is not None or (
                    nxt_text.isupper() and len(nxt_text) > 5 and "\n" not in nxt_text
                )
                if is_next_title:
                    break
                if nxt[6] == 0 and nxt_text:
                    section_blocks.append(nxt)
                    consumed.add(j)

            chunker.emit_external(self._section_chunk(page, number, doc_id, matched, section_blocks))

        chunker.flush()
        return chunker.chunks

    def _section_chunk(self, page, number: int, doc_id: str, matched: str, section_blocks: list) -> RawChunk:
        x0 = min(b[0] for b in section_blocks)
        y0 = min(b[1] for b in section_blocks)
        x1 = max(b[2] for b in section_blocks)
        y1 = max(b[3] for b in section_blocks)
        clip = pymupdf.Rect(
            max(0, x0 - 10),
            max(0, y0 - 10),
            min(page.rect.width, x1 + 10),
            min(page.rect.height, y1 + 10),
        )

        image_path: Optional[str] = None
        try:
            safe = re.sub(r"\W+", "_", matched.lower()).strip("_")
            path = self.output_dir / f"{doc_id}_p{number}_{safe}.png"
            page.get_pixmap(clip=clip, dpi=self.dpi).save(str(path))
            image_path = str(path)
        except Exception as exc:
            logger.warning("Capture impossible (%s, page %s) : %s", matched, number, exc)

        # Tableaux de la section → Markdown ; le texte des cellules n'est pas répété en vrac.
        table_boxes: list[tuple[float, float, float, float]] = []
        markdown_parts: list[str] = []
        try:
            for tab in page.find_tables(clip=clip).tables:
                rows = normalize_rows(tab.extract())
                if len(rows) >= 2 and len(rows[0]) >= 2:
                    table_boxes.append(tuple(tab.bbox))  # type: ignore[arg-type]
                    markdown_parts.extend(rows_to_markdown_chunks(rows, 10**6))
        except Exception as exc:
            logger.debug("find_tables indisponible sur la section %s : %s", matched, exc)

        free_text = "\n".join(
            b[4].strip()
            for b in section_blocks
            if not any(overlap_ratio(tuple(b[:4]), tb) > 0.5 for tb in table_boxes)  # type: ignore[arg-type]
        )
        text = free_text + ("\n\n" + "\n\n".join(markdown_parts) if markdown_parts else "")

        return RawChunk(
            kind="table",
            text=text,
            page_start=number,
            page_end=number,
            bbox=(clip.x0, clip.y0, clip.x1, clip.y1),
            section=matched,
            image_path=image_path,
            needs_image=image_path is not None,  # sections KID complexes : la capture reste utile au VLM
        )
