"""Pure-python (brute-force cosine) vector index adapter.

Used when FAISS is unavailable or VECTOR_BACKEND=local. Fully offline.
"""

from __future__ import annotations

from domain.ports import VectorIndex

from .base import BaseVectorIndex


class LocalVectorIndex(BaseVectorIndex):
    """Brute-force cosine-similarity vector index (no vendor dependency)."""

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
