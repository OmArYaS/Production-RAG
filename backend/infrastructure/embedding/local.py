"""Local dedicated embedding adapter.

Prefers ``fastembed`` then ``sentence-transformers`` (default
``BAAI/bge-small-en-v1.5``). Falls back to a deterministic hashed embedding
when neither library is available, so the app starts and tests run fully
offline. The chat LLM is NEVER used for embedding.
"""

from __future__ import annotations

import hashlib
from typing import List

from domain.ports import EmbeddingProvider


class LocalEmbeddingAdapter(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._backend: str = "offline-fallback"
        self._model = None
        self._dim: int = 384
        self._init_backend()

    def _init_backend(self) -> None:
        # Try fastembed first.
        try:
            from fastembed import TextEmbedding  # type: ignore

            self._model = TextEmbedding(model_name=self._model_name)
            probe = list(self._model.embed(["dimension-probe"]))[0]
            self._dim = len(probe)
            self._backend = "fastembed"
            return
        except Exception:
            self._model = None

        # Try sentence-transformers next.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self._model_name)
            probe = self._model.encode(["dimension-probe"])
            self._dim = int(len(probe[0]))
            self._backend = "sentence-transformers"
            return
        except Exception:
            self._model = None

        # Offline deterministic fallback (no network, no model download).
        self._backend = "offline-fallback"
        self._dim = 384

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._backend == "fastembed":
            return [list(map(float, v)) for v in self._model.embed(texts)]
        if self._backend == "sentence-transformers":
            return self._model.encode(texts, normalize_embeddings=True).tolist()
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def dimension(self) -> int:
        return self._dim

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model_name(self) -> str:
        return self._model_name

    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic hashed embedding (offline fallback)."""
        vec = [0.0] * self._dim
        for word in text.lower().split():
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if ((h // self._dim) % 2 == 0) else -1.0
            vec[idx] += sign
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]
