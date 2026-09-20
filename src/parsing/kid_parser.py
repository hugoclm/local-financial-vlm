from pathlib import Path
import pymupdf
from src.parsing.base import BaseDocumentParser
from src.parsing.schemas import DocumentChunk, ChunkType


class KIDParser(BaseDocumentParser):
    """Parseur haute fidélité pour DIC / KID PRIIPs (3-4 pages)."""

    TARGET_SECTIONS = [
        "INDICATEUR DE RISQUE",
        "SCÉNARIOS DE PERFORMANCE",
        "COÛTS AU FIL DU TEMPS",
        "COMPOSITION DES COÛTS",
    ]

    def extract_chunks(self, pdf_path: str) -> list[DocumentChunk]:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

        doc = pymupdf.open(pdf_file)
        chunks: list[DocumentChunk] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: b[1])  # Tri vertical

            consumed_indices = set()

            for i, block in enumerate(blocks):
                if i in consumed_indices or block[6] != 0:
                    continue

                text = block[4].strip()
                bbox = (block[0], block[1], block[2], block[3])

                if not text or (text.startswith("Page ") and len(text) < 15):
                    continue

                matched_section = None
                for sec in self.TARGET_SECTIONS:
                    if sec in text.upper():
                        matched_section = sec
                        break

                if matched_section:
                    section_blocks = [block]
                    consumed_indices.add(i)

                    next_idx = i + 1
                    while next_idx < len(blocks):
                        nxt = blocks[next_idx]
                        nxt_text = nxt[4].strip()
                        if any(sec in nxt_text.upper() for sec in self.TARGET_SECTIONS) or (
                            nxt_text.isupper() and len(nxt_text) > 5 and "\n" not in nxt_text
                        ):
                            break
                        if nxt[6] == 0 and nxt_text:
                            section_blocks.append(nxt)
                            consumed_indices.add(next_idx)
                        next_idx += 1

                    x0 = min(b[0] for b in section_blocks)
                    y0 = min(b[1] for b in section_blocks)
                    x1 = max(b[2] for b in section_blocks)
                    y1 = max(b[3] for b in section_blocks)

                    clip_rect = pymupdf.Rect(
                        max(0, x0 - 10),
                        max(0, y0 - 10),
                        min(page.rect.width, x1 + 10),
                        min(page.rect.height, y1 + 10),
                    )

                    pix = page.get_pixmap(clip=clip_rect, dpi=150)
                    safe_name = matched_section.lower().replace(" ", "_")
                    img_name = f"{pdf_file.stem}_p{page_number}_{safe_name}.png"
                    img_path = self.output_dir / img_name
                    pix.save(str(img_path))

                    combined_text = "\n".join(b[4].strip() for b in section_blocks)

                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{pdf_file.stem}_p{page_number}_{safe_name}",
                            source_document=pdf_file.name,
                            page_number=page_number,
                            chunk_type=ChunkType.TABLE,
                            text_content=combined_text,
                            image_path=str(img_path),
                            bbox=(clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1),
                        )
                    )
                else:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{pdf_file.stem}_p{page_number}_b{i}",
                            source_document=pdf_file.name,
                            page_number=page_number,
                            chunk_type=ChunkType.TEXT,
                            text_content=text,
                            image_path=None,
                            bbox=bbox,
                        )
                    )

        doc.close()
        return chunks