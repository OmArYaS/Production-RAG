"""Vector index adapters (dense semantic search).

Prefers FAISS when available; falls back to a pure-python cosine-similarity
index so the app starts and tests run fully offline. Supports dimension
safety and doc-aware deletion (no orphan vectors).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from domain.exceptions import EmbeddingDimensionError
from domain.ports import VectorIndex


class _VectorIndexState:
    """Shared in-memory state for vector index adapters."""

    def __init__(self, dimension: int) -> None:
        self._dim = dimension
        self._vectors: Dict[str, List[float]] = {}  # chunk_id -> vector
        self._doc_chunks: Dict[str, Set[str]] = {}  # doc_id -> set of chunk_ids

    @property
    def dimension(self) -> int:
        return self._dim

    def _add_vector(self, chunk_id: str, vector: List[float], doc_id: str) -> None:
        if len(vector) != self._dim:
            raise EmbeddingDimensionError(expected=self._dim, actual=len(vector))
        self._vectors[chunk_id] = list(map(float, vector))
        self._doc_chunks.setdefault(doc_id, set()).add(chunk_id)

    def _chunk_ids_for_doc(self, doc_id: str) -> Set[str]:
        return set(self._doc_chunks.get(doc_id, set()))

    def _remove_chunk(self, chunk_id: str) -> Optional[str]:
        """Remove a single chunk; return its doc_id or None."""
        doc_id = None
        for did, chunks in self._doc_chunks.items():
            if chunk_id in chunks:
                chunks.discard(chunk_id)
                doc_id = did
                if not chunks:
                    del self._doc_chunks[did]
                break
        self._vectors.pop(chunk_id, None)
        return doc_id

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5 or 1.0
        nb = sum(y * y for y in b) ** 0.5 or 1.0
        return dot / (na * nb)


class BaseVectorIndex(VectorIndex, _VectorIndexState):
    def __init__(self, dimension: int) -> None:
        _VectorIndexState.__init__(self, dimension)

    def dimension(self) -> int:
        return self._dim

    def add(
        self, items: List[Tuple[str, str, List[float]]]
    ) -> None:
        """Add vectors as (chunk_id, doc_id, vector) tuples."""
        for chunk_id, doc_id, vector in items:
            self._add_vector(chunk_id, vector, doc_id)

    def search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[str, float]]:
        if len(query_vector) != self._dim:
            raise EmbeddingDimensionError(expected=self._dim, actual=len(query_vector))
        if not self._vectors:
            return []
        scored = [
            (cid, self._cosine(query_vector, v)) for cid, v in self._vectors.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def delete(self, doc_id: str) -> int:
        chunk_ids = list(self._chunk_ids_for_doc(doc_id))
        for cid in chunk_ids:
            self._remove_chunk(cid)
        return len(chunk_ids)

    def reset(self) -> None:
        self._vectors.clear()
        self._doc_chunks.clear()
