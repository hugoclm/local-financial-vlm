from abc import ABC, abstractmethod
from pathlib import Path
from src.parsing.schemas import DocumentChunk


class BaseDocumentParser(ABC):
    """Interface abstraite commune à tous les extracteurs de documents financiers."""

    def __init__(self, output_dir: str = "data/cache/extracted_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract_chunks(self, pdf_path: str) -> list[DocumentChunk]:
        """Extrait l'ensemble des chunks (texte et images de support) d'un document."""
        pass