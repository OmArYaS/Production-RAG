"""Shared fake/mock adapters for deterministic offline tests.

These implement the ``domain/`` ports, so application/presentation use cases can
be exercised without any real infra (no Ollama, no network, no model downloads).
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Dict, List, Optional, Tuple

from domain.models import ChunkRecord, DocumentRecord, QueryLogEntry, SearchHit
from domain.ports import (
    DocumentRepository,
    EmbeddingProvider,
    KeywordIndex,
    LLMProvider,
    VectorIndex,
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeEmbedding(EmbeddingProvider):
    """Deterministic hashed-bag-of-words embedding."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9]+", text.lower())]

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self._dim
        for tok in self._tokens(text):
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            v[h % self._dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._vec(query)

    def dimension(self) -> int:
        return self._dim


class FakeVectorIndex(VectorIndex):
    """Controllable fake vector index.

    ``search_results`` is a mapping doc_id -> ranked [(chunk_id, score)] used
    to simulate dense retrieval. If not set, brute-force cosine over stored
    vectors is used.
    """

    def __init__(self, dimension: int = 8, search_results=None):
        self._dim = dimension
        self._vectors: Dict[str, Tuple[str, List[float]]] = {}  # chunk_id -> (doc_id, vector)
        self.search_results = search_results  # {doc_id?} unused; see search()
        self.calls: List[str] = []

    def add(self, items: List[Tuple[str, str, List[float]]]) -> None:
        for chunk_id, doc_id, vector in items:
            self._vectors[chunk_id] = (doc_id, vector)

    def search(self, query_vector: List[float], top_k: int) -> List[Tuple[str, float]]:
        self.calls.append("search")
        if self.search_results is not None:
            return list(self.search_results)[:top_k]
        # Brute-force cosine for realism.
        scored = [
            (cid, self._cos(query_vector, v)) for cid, (_d, v) in self._vectors.items()
        ]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_k]

    def delete(self, doc_id: str) -> int:
        ids = [cid for cid, (d, _v) in self._vectors.items() if d == doc_id]
        for cid in ids:
            del self._vectors[cid]
        return len(ids)

    def reset(self) -> None:
        self._vectors.clear()

    def dimension(self) -> int:
        return self._dim

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5 or 1.0
        nb = sum(y * y for y in b) ** 0.5 or 1.0
        return dot / (na * nb)


class FakeKeywordIndex(KeywordIndex):
    """Controllable fake keyword index; `search_results` overrides search()."""

    def __init__(self, search_results=None):
        self._chunks: Dict[str, ChunkRecord] = {}
        self._doc_chunks: Dict[str, List[str]] = {}
        self.search_results = search_results
        self.calls: List[str] = []

    def add(self, chunks: List[ChunkRecord]) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c
            self._doc_chunks.setdefault(c.doc_id, []).append(c.chunk_id)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        self.calls.append("search")
        if self.search_results is not None:
            return list(self.search_results)[:top_k]
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = [
            (cid, float(len(tokens & set(re.findall(r"[a-z0-9]+", c.text.lower())))))
            for cid, c in self._chunks.items()
        ]
        scored = [(cid, s) for cid, s in scored if s > 0]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_k]

    def delete(self, doc_id: str) -> int:
        ids = self._doc_chunks.pop(doc_id, [])
        for cid in ids:
            self._chunks.pop(cid, None)
        return len(ids)

    def reset(self) -> None:
        self._chunks.clear()
        self._doc_chunks.clear()


class FakeDocumentRepository(DocumentRepository):
    def __init__(self):
        self._docs: Dict[str, DocumentRecord] = {}
        self._chunks: Dict[str, ChunkRecord] = {}

    def save_document(self, doc: DocumentRecord) -> None:
        self._docs[doc.doc_id] = doc

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        return self._docs.get(doc_id)

    def list_documents(self) -> List[DocumentRecord]:
        return list(self._docs.values())

    def delete_document(self, doc_id: str) -> bool:
        return self._docs.pop(doc_id, None) is not None

    def save_chunk(self, chunk: ChunkRecord) -> None:
        self._chunks[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        return self._chunks.get(chunk_id)

    def get_chunks_by_document(self, doc_id: str) -> List[ChunkRecord]:
        return [c for c in self._chunks.values() if c.doc_id == doc_id]

    def delete_chunks_by_document(self, doc_id: str) -> int:
        ids = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in ids:
            del self._chunks[cid]
        return len(ids)


class FakeQueryLogRepository:
    """Minimal in-memory query log (implements QueryLogRepository)."""

    def __init__(self):
        self._entries: List[QueryLogEntry] = []

    def save_entry(self, entry: QueryLogEntry) -> None:
        self._entries.append(entry)

    def list_entries(self, limit: int = 100) -> List[QueryLogEntry]:
        return list(reversed(self._entries))[:limit]


class FakeLLM(LLMProvider):
    """Fake chat LLM returning canned text (supports think tags/fences)."""

    def __init__(self, canned: str = "Answer from context [1]."):
        self.canned = canned
        self.prompts: List[Tuple[str, str]] = []
        self._model = "fake-model"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        time.sleep(0.001)
        return self.canned
