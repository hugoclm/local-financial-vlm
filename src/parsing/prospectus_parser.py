from pathlib import Path
import pymupdf
from src.parsing.base import BaseDocumentParser
from src.parsing.schemas import DocumentChunk, ChunkType


class FullProspectusParser(BaseDocumentParser):
    """Parseur universel et sans hypothèse de titrage pour prospectus longs."""

    def __init__(self, output_dir: str = "data/cache/extracted_images", chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text_with_overlap(self, text: str) -> list[str]:
        """Découpe un texte par fenêtre glissante en respectant au mieux les sauts de ligne."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{p}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Conserve la fin du chunk précédent pour le recouvrement
                    current_chunk = current_chunk[-self.chunk_overlap :] + "\n\n" + p
                else:
                    chunks.append(p[: self.chunk_size])
                    current_chunk = p[self.chunk_size - self.chunk_overlap :]

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def extract_chunks(self, pdf_path: str) -> list[DocumentChunk]:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

        doc = pymupdf.open(pdf_file)
        chunks: list[DocumentChunk] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1

            # 1. Capture des tableaux et grilles vectorielles
            tabs = page.find_tables()
            table_bboxes = []
            for tab_idx, table in enumerate(tabs):
                bbox = table.bbox
                table_bboxes.append(bbox)
                rect = pymupdf.Rect(bbox)

                tab_text = page.get_text("text", clip=rect).strip()
                if not tab_text or len(tab_text) < 25:
                    continue

                pix = page.get_pixmap(clip=rect, dpi=150)
                img_name = f"{pdf_file.stem}_p{page_number}_tab{tab_idx}.png"
                img_path = self.output_dir / img_name
                pix.save(str(img_path))

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{pdf_file.stem}_p{page_number}_tab_{tab_idx}",
                        source_document=pdf_file.name,
                        page_number=page_number,
                        chunk_type=ChunkType.TABLE,
                        text_content=tab_text,
                        image_path=str(img_path),
                        bbox=bbox,
                    )
                )

            # 2. Extraction du texte de la page (en masquant les zones déjà prises en tableau)
            text_page = page.get_text("text").strip()
            if not text_page or len(text_page) < 30:
                continue

            text_splits = self._split_text_with_overlap(text_page)
            for split_idx, split_content in enumerate(text_splits):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{pdf_file.stem}_p{page_number}_chunk_{split_idx}",
                        source_document=pdf_file.name,
                        page_number=page_number,
                        chunk_type=ChunkType.TEXT,
                        text_content=split_content,
                        image_path=None,
                        bbox=(0, 0, page.rect.width, page.rect.height),
                    )
                )

        doc.close()
        return chunks