"""Ressources lourdes partagées (une seule instance par processus).

Avant : l'indexeur ET le retriever chargeaient chacun BGE-M3, et l'application Streamlit
rechargeait tout à chaque clic. Ici chaque modèle / client Qdrant est créé une seule fois.
Les imports lourds (torch, sentence-transformers, qdrant) sont différés au premier usage.
"""
from __future__ import annotations

import atexit
import logging
import threading
from typing import Any, Callable

from src.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_cache: dict[Any, Any] = {}
_index_versions: dict[str, int] = {}


def _cached(key: Any, factory: Callable[[], Any]) -> Any:
    with _lock:
        if key not in _cache:
            _cache[key] = factory()
        return _cache[key]


def resolve_device(preference: str) -> str:
    if preference and preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # torch absent ou cassé
        pass
    return "cpu"


def get_encoder(model_name: str | None = None, device: str | None = None):
    settings = get_settings()
    model_name = model_name or settings.embedding_model
    device = resolve_device(device or settings.embedding_device)

    def load():
        from sentence_transformers import SentenceTransformer

        logger.info("Chargement de l'encodeur %s sur %s", model_name, device)
        encoder = SentenceTransformer(model_name, device=device)
        encoder.max_seq_length = settings.max_seq_length  # BGE-M3 vaut 8192 par défaut : inutile ici
        if device.startswith("cuda"):
            encoder.half()
        return encoder

    return _cached(("encoder", model_name, device), load)


def get_reranker(model_name: str | None = None, device: str | None = None):
    settings = get_settings()
    model_name = model_name or settings.reranker_model
    device = resolve_device(device or settings.reranker_device)

    def load():
        from sentence_transformers import CrossEncoder

        logger.info("Chargement du reranker %s sur %s", model_name, device)
        reranker = CrossEncoder(model_name, device=device, max_length=settings.rerank_max_length)
        if device.startswith("cuda"):
            reranker.model.half()
        return reranker

    return _cached(("reranker", model_name, device), load)


def get_client(storage_path: str | None = None, url: str | None = None):
    """Client Qdrant unique par cible (le mode embarqué verrouille le dossier de stockage)."""
    settings = get_settings()
    url = url or settings.qdrant_url
    storage_path = storage_path or settings.storage_path

    def load():
        from qdrant_client import QdrantClient

        if url:
            logger.info("Connexion au serveur Qdrant %s", url)
            return QdrantClient(url=url)
        logger.info("Ouverture de Qdrant embarqué : %s", storage_path)
        return QdrantClient(path=str(storage_path))

    return _cached(("client", url or str(storage_path)), load)


def embedding_dimension(encoder) -> int:
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        fn = getattr(encoder, attr, None)
        if fn is not None:
            dim = fn()
            if dim:
                return int(dim)
    return int(encoder.encode("dimension", show_progress_bar=False).shape[-1])


def bump_index_version(collection: str) -> None:
    """Invalide les caches lexicaux d'une collection après une écriture."""
    with _lock:
        _index_versions[collection] = _index_versions.get(collection, 0) + 1


def index_version(collection: str) -> int:
    return _index_versions.get(collection, 0)


def close_clients() -> None:
    with _lock:
        for key, value in list(_cache.items()):
            if isinstance(key, tuple) and key and key[0] == "client":
                try:
                    value.close()
                except Exception:
                    pass
                _cache.pop(key, None)


atexit.register(close_clients)
