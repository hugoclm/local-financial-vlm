from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class DocumentChunk(BaseModel):
    """Représente une unité d'information extraite du PDF."""

    chunk_id: str = Field(description="Identifiant unique (doc_id + rang du chunk)")
    doc_id: str = Field(default="", description="Empreinte du fichier : sépare les documents dans la collection")
    source_document: str = Field(description="Nom du document d'origine")
    page_number: int = Field(description="Page de début dans le PDF (base 1)")
    page_end: Optional[int] = Field(default=None, description="Page de fin si le chunk chevauche plusieurs pages")
    chunk_type: ChunkType = Field(description="Nature du contenu")
    section: Optional[str] = Field(default=None, description="Fil d'Ariane des titres (ex. « Objectifs > Limites »)")
    chunk_index: int = Field(default=0, description="Rang du chunk dans le document (sert aux voisins)")
    text_content: Optional[str] = Field(default=None, description="Contenu textuel (Markdown pour les tableaux)")
    image_path: Optional[str] = Field(default=None, description="Chemin de l'image pour tableaux/figures")
    needs_image: bool = Field(default=False, description="Vrai si le texte extrait est douteux : envoyer l'image au VLM")
    bbox: tuple[float, float, float, float] = Field(description="Coordonnées de l'élément (x0, y0, x1, y1)")

    @property
    def retrieval_text(self) -> str:
        """Texte réellement vectorisé et reranké : le titre de section désambiguïse les textes juridiques répétitifs."""
        body = (self.text_content or "").strip() or f"Tableau financier extrait de la page {self.page_number}"
        return f"{self.section}\n{body}" if self.section else body
