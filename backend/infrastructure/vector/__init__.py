"""Vector index adapters (dense semantic search)."""

from typing import List, Tuple

from domain.ports import VectorIndex

from .faiss import FAISSVectorIndex, _faiss_available
from .local import LocalVectorIndex


def build_vector_index(settings, dimension: int) -> VectorIndex:
    """Factory: build the configured vector index.

    ``VECTOR_BACKEND=auto`` (default) uses FAISS when available, else the
    pure-python cosine index. Requires only changing DI wiring to swap.
    """
    backend = (settings.VECTOR_BACKEND or "auto").lower()
    want_faiss = backend == "faiss" or (
        backend == "auto" and _faiss_available()
    )
    if want_faiss:
        try:
            return FAISSVectorIndex(dimension)
        except Exception:
            # Fall through to the local brute-force index.
            pass
    return LocalVectorIndex(dimension)
