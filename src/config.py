"""Configuration centralisée.

Toutes les valeurs peuvent être surchargées par variables d'environnement, par exemple :

    $env:EMBEDDING_DEVICE = "cuda"      # PowerShell
    $env:QDRANT_URL = "http://localhost:6333"

Les défauts conservent le comportement d'origine du projet (embeddings et reranker sur CPU
pour laisser toute la VRAM à Ollama).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _s(name: str, default: str) -> str:
    return os.getenv(name, default)


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "oui"}


@dataclass(frozen=True)
class Settings:
    # --- Stockage ---------------------------------------------------------------------
    storage_path: str = field(default_factory=lambda: _s("QDRANT_PATH", "qdrant_storage"))
    # Si défini, on utilise un serveur Qdrant (Docker) au lieu du mode embarqué.
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("QDRANT_URL") or None)
    image_dir: str = field(default_factory=lambda: _s("IMAGE_DIR", "data/cache/extracted_images"))
    samples_dir: str = field(default_factory=lambda: _s("SAMPLES_DIR", "data/samples"))

    # --- Modèles d'embedding / reranking ---------------------------------------------
    embedding_model: str = field(default_factory=lambda: _s("EMBEDDING_MODEL", "BAAI/bge-m3"))
    reranker_model: str = field(default_factory=lambda: _s("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    embedding_device: str = field(default_factory=lambda: _s("EMBEDDING_DEVICE", "cpu"))  # cpu|cuda|mps|auto
    reranker_device: str = field(default_factory=lambda: _s("RERANKER_DEVICE", "cpu"))
    encode_batch_size: int = field(default_factory=lambda: _i("ENCODE_BATCH_SIZE", 16))
    rerank_batch_size: int = field(default_factory=lambda: _i("RERANK_BATCH_SIZE", 16))
    max_seq_length: int = field(default_factory=lambda: _i("MAX_SEQ_LENGTH", 512))
    rerank_max_length: int = field(default_factory=lambda: _i("RERANK_MAX_LENGTH", 512))

    # --- Parsing / chunking ----------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _i("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _i("CHUNK_OVERLAP", 150))
    table_chunk_size: int = field(default_factory=lambda: _i("TABLE_CHUNK_SIZE", 2500))
    min_chunk_chars: int = field(default_factory=lambda: _i("MIN_CHUNK_CHARS", 250))
    table_dpi: int = field(default_factory=lambda: _i("TABLE_DPI", 150))

    # --- Retrieval -------------------------------------------------------------------
    hybrid: bool = field(default_factory=lambda: _b("HYBRID_SEARCH", True))
    rrf_k: int = field(default_factory=lambda: _i("RRF_K", 60))
    decompose_mode: str = field(default_factory=lambda: _s("DECOMPOSE_MODE", "rules"))  # off|rules|llm
    neighbor_window: int = field(default_factory=lambda: _i("NEIGHBOR_WINDOW", 1))
    max_context_chars: int = field(default_factory=lambda: _i("MAX_CONTEXT_CHARS", 9000))

    # --- Génération (Ollama) ---------------------------------------------------------
    ollama_model: str = field(default_factory=lambda: _s("VLM_MODEL", "qwen2.5vl:7b"))
    # IMPORTANT : garder num_ctx constant entre tous les appels (y compris la planification
    # de requête) : Ollama recharge le modèle quand la taille de contexte change.
    num_ctx: int = field(default_factory=lambda: _i("OLLAMA_NUM_CTX", 8192))
    num_predict: int = field(default_factory=lambda: _i("OLLAMA_NUM_PREDICT", 1024))
    keep_alive: str = field(default_factory=lambda: _s("OLLAMA_KEEP_ALIVE", "30m"))
    temperature: float = field(default_factory=lambda: _f("VLM_TEMPERATURE", 0.0))
    max_images: int = field(default_factory=lambda: _i("MAX_IMAGES", 2))
    max_image_side: int = field(default_factory=lambda: _i("MAX_IMAGE_SIDE", 1280))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
