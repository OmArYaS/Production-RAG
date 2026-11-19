"""Application configuration loaded from environment / ``.env``.

Zero-cost local stack. All values have sensible defaults so the app starts
without any cloud credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


def _getenv(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _getenv_bool(key: str, default: bool = False) -> bool:
    return _getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _getenv_int(key: str, default: int) -> int:
    try:
        return int(_getenv(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Embedding (dedicated local model, never the chat LLM) ---
    EMBEDDING_PROVIDER: str = "local"  # "local" | "ollama"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    # Ollama embedding model (used when EMBEDDING_PROVIDER=ollama)
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # --- Chat LLM (model-agnostic local Ollama) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct-q4_k_m"

    # --- Vector / keyword index ---
    VECTOR_INDEX_PATH: str = ".data/vector_index.json"
    VECTOR_BACKEND: str = "auto"  # "auto" | "faiss" | "local"

    # --- SQLite persistence ---
    DATABASE_URL: str = "sqlite:///.data/rag.db"

    # --- Chunking ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # --- Retrieval ---
    TOP_K: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            EMBEDDING_PROVIDER=_getenv("EMBEDDING_PROVIDER", "local"),
            EMBEDDING_MODEL=_getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            OLLAMA_EMBED_MODEL=_getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            OLLAMA_BASE_URL=_getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            OLLAMA_MODEL=_getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_k_m"),
            VECTOR_INDEX_PATH=_getenv("VECTOR_INDEX_PATH", ".data/vector_index.json"),
            VECTOR_BACKEND=_getenv("VECTOR_BACKEND", "auto"),
            DATABASE_URL=_getenv("DATABASE_URL", "sqlite:///.data/rag.db"),
            CHUNK_SIZE=_getenv_int("CHUNK_SIZE", 500),
            CHUNK_OVERLAP=_getenv_int("CHUNK_OVERLAP", 50),
            TOP_K=_getenv_int("TOP_K", 10),
        )
