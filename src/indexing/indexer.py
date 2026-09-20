from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.parsing.schemas import DocumentChunk


class LocalVectorIndexer:
    """Gère la vectorisation sémantique et l'indexation dans Qdrant local."""

    def __init__(
        self,
        collection_name: str = "financial_docs",
        storage_path: str = "qdrant_storage",
        model_name: str = "BAAI/bge-m3",
    ):
        self.collection_name = collection_name
        self.storage_path = Path(storage_path)

        # 1. Chargement du modèle d'embeddings local
        print(f"Chargement du modèle d'embeddings ({model_name})...")
        self.encoder = SentenceTransformer(model_name, device = "cpu")
        # BGE-M3 produit des vecteurs de dimension 1024
        self.vector_dim = self.encoder.get_embedding_dimension()

        # 2. Initialisation de Qdrant en local direct sur disque
        self.client = QdrantClient(path=str(self.storage_path))
        self._ensure_collection()

    def _ensure_collection(self):
        """Crée l'index dans Qdrant s'il n'existe pas encore."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                ),
            )
            print(f"Collection '{self.collection_name}' initialisée (dimension {self.vector_dim}).")

    def close(self):
        """Ferme la connexion locale et libère le verrou de stockage."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()

    def index_chunks(self, chunks: list[DocumentChunk]):
        """Calcule les vecteurs et les enregistre dans Qdrant avec leurs métadonnées."""
        if not chunks:
            print("Aucun chunk à indexer.")
            return

        print(f"Vectorisation de {len(chunks)} chunks...")

        # Préparation du contenu textuel pour chaque chunk
        texts_to_embed = [
            chunk.text_content if chunk.text_content else f"Tableau financier extrait de la page {chunk.page_number}"
            for chunk in chunks
        ]

        # Calcul matriciel des embeddings
        embeddings = self.encoder.encode(texts_to_embed, show_progress_bar=True)

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "source_document": chunk.source_document,
                        "page_number": chunk.page_number,
                        "chunk_type": chunk.chunk_type.value,
                        "text_content": chunk.text_content,
                        "image_path": chunk.image_path,
                        "bbox": list(chunk.bbox),
                    },
                )
            )

        # Enregistrement par lot dans la base
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"Indexation terminée : {len(points)} vecteurs stockés dans Qdrant.")