from __future__ import annotations

import logging
import uuid
from typing import Callable, Optional

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from src.config import Settings, get_settings
from src.models import bump_index_version, embedding_dimension, get_client, get_encoder
from src.parsing.schemas import DocumentChunk

logger = logging.getLogger(__name__)


def _doc_filter(doc_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])


class LocalVectorIndexer:
    """Vectorisation sémantique et indexation dans Qdrant.

    Corrections par rapport à la version d'origine :
    - identifiants de points déterministes (uuid5) : plus d'écrasement entre documents (id=0..n) ;
    - `doc_id` dans le payload : on peut filtrer, supprimer ou ré-indexer un seul document ;
    - encodeur et client Qdrant partagés (voir src/models.py) ;
    - vecteurs normalisés, batches triés par longueur, progression rapportée à l'appelant.
    """

    def __init__(
        self,
        collection_name: str = "financial_docs",
        storage_path: Optional[str] = None,
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.encoder = get_encoder(model_name)
        self.client = get_client(storage_path)
        self.vector_dim = embedding_dimension(self.encoder)
        self._ensure_collection()

    # ------------------------------------------------------------ collection
    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )
        # Index de payload : utiles avec un serveur Qdrant ; ignorés ou sans effet en mode embarqué.
        for field, schema in (("doc_id", PayloadSchemaType.KEYWORD), ("chunk_index", PayloadSchemaType.INTEGER)):
            try:
                self.client.create_payload_index(self.collection_name, field_name=field, field_schema=schema)
            except Exception as exc:
                logger.debug("Index de payload '%s' non créé : %s", field, exc)
        logger.info("Collection '%s' initialisée (dimension %d).", self.collection_name, self.vector_dim)

    def close(self) -> None:
        """Conservé pour compatibilité : le client est partagé et fermé à la sortie du processus."""

    # ------------------------------------------------------------- documents
    def is_indexed(self, doc_id: str) -> bool:
        return self.client.count(self.collection_name, count_filter=_doc_filter(doc_id), exact=True).count > 0

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(self.collection_name, points_selector=FilterSelector(filter=_doc_filter(doc_id)))
        bump_index_version(self.collection_name)

    def list_documents(self) -> list[dict]:
        """Documents présents dans la collection : [{doc_id, source_document, chunks}]."""
        docs: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection_name,
                limit=512,
                offset=offset,
                with_payload=["doc_id", "source_document"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id") or "(ancien index)"
                entry = docs.setdefault(
                    doc_id, {"doc_id": doc_id, "source_document": payload.get("source_document", "?"), "chunks": 0}
                )
                entry["chunks"] += 1
            if offset is None:
                break
        return sorted(docs.values(), key=lambda d: d["source_document"])

    # -------------------------------------------------------------- indexation
    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        replace: bool = True,
        progress: Optional[Callable[[float], None]] = None,
    ) -> int:
        """Vectorise et enregistre les chunks. `replace=True` supprime d'abord l'ancienne version du document."""
        if not chunks:
            logger.warning("Aucun chunk à indexer.")
            return 0

        if replace:
            for doc_id in {c.doc_id for c in chunks if c.doc_id}:
                self.delete_document(doc_id)

        texts = [c.retrieval_text for c in chunks]
        vectors = self._encode(texts, progress)

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_document": chunk.source_document,
                    "page_number": chunk.page_number,
                    "page_end": chunk.page_end,
                    "chunk_type": chunk.chunk_type.value,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "text_content": chunk.text_content,
                    "retrieval_text": chunk.retrieval_text,
                    "image_path": chunk.image_path,
                    "needs_image": chunk.needs_image,
                    "bbox": list(chunk.bbox),
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        for start in range(0, len(points), 256):
            self.client.upsert(self.collection_name, points=points[start : start + 256], wait=True)

        bump_index_version(self.collection_name)
        logger.info("Indexation terminée : %d vecteurs dans '%s'.", len(points), self.collection_name)
        return len(points)

    def _encode(self, texts: list[str], progress: Optional[Callable[[float], None]]) -> list[list[float]]:
        """Encode par tranches triées par longueur (moins de padding), puis remet dans l'ordre d'origine."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors: list[Optional[list[float]]] = [None] * len(texts)
        step = max(self.settings.encode_batch_size * 4, 16)
        for start in range(0, len(order), step):
            idx = order[start : start + step]
            batch = self.encoder.encode(
                [texts[i] for i in idx],
                batch_size=self.settings.encode_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for i, vec in zip(idx, batch):
                vectors[i] = vec.tolist()
            if progress:
                progress(min(1.0, (start + len(idx)) / len(texts)))
        return vectors  # type: ignore[return-value]
