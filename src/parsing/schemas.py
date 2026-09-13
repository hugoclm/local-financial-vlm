from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class DocumentChunk(BaseModel):
    """Représente une unité d'information extraite du PDF."""
    chunk_id: str = Field(description="Identifiant unique du chunk")
    page_number: int = Field(description="Page d'origine (base 1)")
    chunk_type: ChunkType = Field(description="Nature du contenu")
    text_content: Optional[str] = Field(default=None, description="Texte brut si applicable")
    image_path: Optional[str] = Field(default=None, description="Chemin vers l'image si c'est un tableau/figure")
    bbox: tuple[float, float, float, float] = Field(description="Coordonnées (x0, y0, x1, y1) sur la page")