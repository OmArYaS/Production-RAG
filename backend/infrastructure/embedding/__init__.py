"""Embedding provider adapters (dedicated local models, never the chat LLM)."""

from domain.ports import EmbeddingProvider

from .local import LocalEmbeddingAdapter
from .ollama import OllamaEmbeddingAdapter

__all__ = ["LocalEmbeddingAdapter", "OllamaEmbeddingAdapter", "build_embedding_provider"]


def build_embedding_provider(settings) -> EmbeddingProvider:
    """Factory: build the configured embedding provider.

    Switching between ``local`` (fastembed/sentence-transformers) and
    ``ollama`` requires only changing ``EMBEDDING_PROVIDER`` in ``.env`` and
    this factory — never domain/application code.
    """
    provider = (settings.EMBEDDING_PROVIDER or "local").strip().lower()
    if provider == "ollama":
        return OllamaEmbeddingAdapter(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBED_MODEL,
        )
    # default: dedicated local embedding model
    return LocalEmbeddingAdapter(
        model_name=settings.EMBEDDING_MODEL,
    )
