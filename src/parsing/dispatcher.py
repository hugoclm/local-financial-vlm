import logging
from pathlib import Path
from typing import Optional

import pymupdf

from src.config import get_settings
from src.parsing.base import BaseDocumentParser
from src.parsing.kid_parser import KIDParser
from src.parsing.prospectus_parser import FullProspectusParser
from src.parsing.schemas import DocumentChunk
from src.utils import file_sha256

logger = logging.getLogger(__name__)


class DocumentDispatcher:
    """Aiguilleur sélectionnant la stratégie de parsing optimale."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or get_settings().image_dir

    def get_parser(self, pdf_path: str) -> BaseDocumentParser:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Fichier introuvable : {pdf_path}")

        doc = pymupdf.open(pdf_file)
        try:
            page_count = len(doc)
            first_pages_text = "".join(doc[i].get_text("text").lower() for i in range(min(2, page_count)))
        finally:
            doc.close()

        # Critères d'identification d'un DIC / KID
        is_short_doc = page_count <= 4
        has_kid_keywords = (
            "informations clés" in first_pages_text
            or "priips" in first_pages_text
            or "scénarios de performance" in first_pages_text
        )

        if is_short_doc and has_kid_keywords:
            logger.info("[Dispatcher] KID / PRIIPs (%d pages) -> KIDParser", page_count)
            return KIDParser(output_dir=self.output_dir)

        logger.info("[Dispatcher] Prospectus complet (%d pages) -> FullProspectusParser", page_count)
        return FullProspectusParser(output_dir=self.output_dir)

    def process(self, pdf_path: str, doc_id: Optional[str] = None) -> list[DocumentChunk]:
        doc_id = doc_id or file_sha256(pdf_path)
        parser = self.get_parser(pdf_path)
        return parser.extract_chunks(pdf_path, doc_id=doc_id)
