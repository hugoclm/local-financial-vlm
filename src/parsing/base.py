from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.config import get_settings
from src.parsing.schemas import DocumentChunk
from src.utils import file_sha256


class BaseDocumentParser(ABC):
    """Interface abstraite commune à tous les extracteurs de documents financiers."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or get_settings().image_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract_chunks(self, pdf_path: str, doc_id: Optional[str] = None) -> list[DocumentChunk]:
        """Extrait l'ensemble des chunks (texte et images de support) d'un document."""

    @staticmethod
    def _resolve_doc_id(pdf_path: str | Path, doc_id: Optional[str]) -> str:
        return doc_id or file_sha256(pdf_path)

    @staticmethod
    def _finalize(chunks: list[DocumentChunk], doc_id: str) -> list[DocumentChunk]:
        """Numérote les chunks dans l'ordre du document et leur attribue un identifiant stable."""
        for index, chunk in enumerate(chunks):
            chunk.doc_id = doc_id
            chunk.chunk_index = index
            chunk.chunk_id = f"{doc_id}_c{index:05d}"
        return chunks
