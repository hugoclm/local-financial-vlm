from pathlib import Path
import pymupdf
from src.parsing.base import BaseDocumentParser
from src.parsing.kid_parser import KIDParser
from src.parsing.prospectus_parser import FullProspectusParser
from src.parsing.schemas import DocumentChunk


class DocumentDispatcher:
    """Aiguilleur intelligent sélectionnant la stratégie de parsing optimale."""

    def __init__(self, output_dir: str = "data/cache/extracted_images"):
        self.output_dir = output_dir

    def get_parser(self, pdf_path: str) -> BaseDocumentParser:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

        doc = pymupdf.open(pdf_file)
        page_count = len(doc)

        # Inspection des premières pages
        first_pages_text = ""
        for i in range(min(2, page_count)):
            first_pages_text += doc[i].get_text("text").lower()
        doc.close()

        # Critères d'identification d'un DIC / KID
        is_short_doc = page_count <= 4
        has_kid_keywords = (
            "informations clés" in first_pages_text
            or "priips" in first_pages_text
            or "scénarios de performance" in first_pages_text
        )

        if is_short_doc and has_kid_keywords:
            print(f"[Dispatcher] Type détecté : KID / PRIIPs ({page_count} pages) -> KIDParser")
            return KIDParser(output_dir=self.output_dir)

        print(f"[Dispatcher] Type détecté : Prospectus complet ({page_count} pages) -> FullProspectusParser")
        return FullProspectusParser(output_dir=self.output_dir)

    def process(self, pdf_path: str) -> list[DocumentChunk]:
        parser = self.get_parser(pdf_path)
        return parser.extract_chunks(pdf_path)