"""FAISS-backed vector index (falls back to brute force if FAISS missing)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from domain.exceptions import EmbeddingDimensionError
from domain.ports import VectorIndex

from .base import BaseVectorIndex, _VectorIndexState


def _faiss_available() -> bool:
    try:
        import faiss  # type: ignore
        return True
    except Exception:
        return False


class FAISSVectorIndex(BaseVectorIndex):
    """Dense vector index using FAISS (IndexFlatIP = cosine on normalized vectors)."""

    def __init__(self, dimension: int) -> None:
        if not _faiss_available():
            raise RuntimeError("FAISS is not installed")
        import faiss  # type: ignore

        _VectorIndexState.__init__(self, dimension)
        self._index = faiss.IndexFlatIP(dimension)
        # faiss internal index -> chunk_id
        self._faiss_id_to_chunk: List[str] = []
        self._chunk_to_faiss_id: dict = {}

    def _rebuild(self) -> None:
        import faiss  # type: ignore

        self._index.reset()
        self._faiss_id_to_chunk = []
        self._chunk_to_faiss_id = {}
        if not self._vectors:
            return
        # Order chunk_ids deterministically.
        chunk_ids = sorted(self._vectors.keys())
        matrix = []
        for new_i, cid in enumerate(chunk_ids):
            vec = np.array(self._vectors[cid], dtype="float32")
            # Normalize so inner product == cosine similarity.
            norm = np.linalg.norm(vec) or 1.0
            matrix.append((vec / norm).astype("float32"))
            self._chunk_to_faiss_id[cid] = new_i
            self._faiss_id_to_chunk.append(cid)
        mat = np.vstack(matrix)
        self._index.add(mat)

    def add(
        self, items: List[Tuple[str, str, List[float]]]
    ) -> None:
        for chunk_id, doc_id, vector in items:
            if len(vector) != self.dimension():
                raise EmbeddingDimensionError(
                    expected=self.dimension(), actual=len(vector)
                )
            self._vectors[chunk_id] = list(map(float, vector))
            self._state_add_doc_chunk(doc_id, chunk_id)
        self._rebuild()

    def _state_add_doc_chunk(self, doc_id: str, chunk_id: str) -> None:
        self._doc_chunks.setdefault(doc_id, set()).add(chunk_id)

    def search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[str, float]]:
        if len(query_vector) != self.dimension():
            raise EmbeddingDimensionError(
                expected=self.dimension(), actual=len(query_vector)
            )
        if not self._vectors:
            return []
        import faiss  # type: ignore

        q = np.array([query_vector], dtype="float32")
        norm = np.linalg.norm(q) or 1.0
        q = (q / norm).astype("float32")
        k = min(top_k, len(self._vectors))
        scores, indices = self._index.search(q, k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._faiss_id_to_chunk):
                continue
            cid = self._faiss_id_to_chunk[idx]
            results.append((cid, float(score)))
        return results

    def delete(self, doc_id: str) -> int:
        chunk_ids = list(self._chunk_ids_for_doc(doc_id))
        for cid in chunk_ids:
            self._remove_chunk(cid)
        self._rebuild()
        return len(chunk_ids)

    def reset(self) -> None:
        _VectorIndexState.reset(self)
        import faiss  # type: ignore

        self._index = faiss.IndexFlatIP(self.dimension())
        self._faiss_id_to_chunk = []
        self._chunk_to_faiss_id = {}
