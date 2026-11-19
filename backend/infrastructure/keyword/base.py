"""Shared base for keyword (sparse) index adapters."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from domain.ports import ChunkRecord, KeywordIndex


class BaseKeywordIndex(KeywordIndex):
    """In-memory storage + abstract BM25 scoring for sparse search."""

    def __init__(self) -> None:
        self._chunks: Dict[str, ChunkRecord] = {}
        self._doc_chunks: Dict[str, Set[str]] = {}
        self._rebuild()

    # -- KeywordIndex contract -----------------------------------------
    def add(self, chunks: List[ChunkRecord]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._doc_chunks.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)
        self._rebuild()

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not query or not query.strip() or not self._chunks:
            return []
        scores = self._score_all(query)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(cid, float(s)) for cid, s in top]

    def delete(self, doc_id: str) -> int:
        chunk_ids = list(self._doc_chunks.get(doc_id, set()))
        for cid in chunk_ids:
            self._chunks.pop(cid, None)
        self._doc_chunks.pop(doc_id, None)
        self._rebuild()
        return len(chunk_ids)

    def reset(self) -> None:
        self._chunks.clear()
        self._doc_chunks.clear()
        self._rebuild()

    # -- Abstract/overridable ------------------------------------------
    def _score_all(self, query: str) -> Dict[str, float]:
        raise NotImplementedError

    def _rebuild(self) -> None:
        pass
