from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class DocumentChunk(BaseModel):
    """Représente une unité d'information extraite du PDF."""
    chunk_id: str = Field(description="Identifiant unique")
    source_document: str = Field(description="Nom du document d'origine")
    page_number: int = Field(description="Page du PDF (base 1)")
    chunk_type: ChunkType = Field(description="Nature du contenu")
    text_content: Optional[str] = Field(default=None, description="Contenu textuel")
    image_path: Optional[str] = Field(default=None, description="Chemin de l'image pour tableaux/figures")
    bbox: tuple[float, float, float, float] = Field(description="Coordonnées de l'élément (x0, y0, x1, y1)")