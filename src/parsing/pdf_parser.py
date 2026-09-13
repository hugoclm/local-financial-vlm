from pathlib import Path
import pymupdf
from src.parsing.schemas import DocumentChunk, ChunkType


class FinancialPDFParser:
    """Parseur hybride optimisé pour les prospectus et rapports financiers."""

    def __init__(self, output_dir: str = "data/cache/extracted_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_chunks(self, pdf_path: str) -> list[DocumentChunk]:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

        doc = pymupdf.open(pdf_file)
        chunks: list[DocumentChunk] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1

            # Détection de tableaux avec stratégie souple (pour tableaux sans bordures nettes)
            tabs = page.find_tables(
                vertical_strategy="text",
                horizontal_strategy="lines_strict"
            )
            
            # Si aucun tableau trouvé, repli sur la stratégie globale
            if len(tabs.tables) == 0:
                tabs = page.find_tables()

            table_bboxes = []

            for tab_idx, table in enumerate(tabs):
                bbox = table.bbox
                table_bboxes.append(bbox)

                # Export d'image haute résolution (DPI 150)
                rect = pymupdf.Rect(bbox)
                pix = page.get_pixmap(clip=rect, dpi=150)
                image_filename = f"{pdf_file.stem}_p{page_number}_tab{tab_idx}.png"
                image_path = self.output_dir / image_filename
                pix.save(str(image_path))

                # Export Markdown
                markdown_table = table.extract()
                table_text = self._format_as_markdown(markdown_table)

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{pdf_file.stem}_p{page_number}_tab_{tab_idx}",
                        source_document=pdf_file.name,
                        page_number=page_number,
                        chunk_type=ChunkType.TABLE,
                        text_content=table_text,
                        image_path=str(image_path),
                        bbox=bbox,
                    )
                )

            # Extraction des blocs textuels narratifs hors zones de tableaux
            blocks = page.get_text("blocks")
            for block_idx, block in enumerate(blocks):
                if block[6] == 0:  # Bloc texte standard
                    bbox = (block[0], block[1], block[2], block[3])
                    text = block[4].strip()

                    if not text or self._is_inside_any_table(bbox, table_bboxes):
                        continue

                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{pdf_file.stem}_p{page_number}_blk_{block_idx}",
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

    def _is_inside_any_table(self, bbox, table_bboxes) -> bool:
        b_rect = pymupdf.Rect(bbox)
        for t_bbox in table_bboxes:
            t_rect = pymupdf.Rect(t_bbox)
            if b_rect.intersects(t_rect):
                return True
        return False

    def _format_as_markdown(self, data: list[list[str]]) -> str:
        if not data or not data[0]:
            return ""
        header = "| " + " | ".join(str(cell or "").replace("\n", " ").strip() for cell in data[0]) + " |"
        separator = "| " + " | ".join("---" for _ in data[0]) + " |"
        rows = [
            "| " + " | ".join(str(cell or "").replace("\n", " ").strip() for cell in row) + " |"
            for row in data[1:]
        ]
        return "\n".join([header, separator] + rows)