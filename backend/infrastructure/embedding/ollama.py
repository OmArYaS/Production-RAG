"""Ollama embedding adapter (nomic-embed-text via /api/embeddings).

Used only when EMBEDDING_PROVIDER=ollama. The chat LLM is never used to embed.
"""

from __future__ import annotations

from typing import List, Optional

import requests

from domain.exceptions import LLMProviderError
from domain.ports import EmbeddingProvider


class OllamaEmbeddingAdapter(EmbeddingProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim: Optional[int] = None
        self._probe_dimension()

    def _probe_dimension(self) -> None:
        try:
            vec = self._embed_one("dimension-probe")
            self._dim = len(vec)
        except Exception:
            # nomic-embed-text is 768-dimensional by default.
            self._dim = 768

    def _embed_one(self, text: str) -> List[float]:
        resp = requests.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=30,
        )
        if resp.status_code >= 400:
            raise LLMProviderError(
                f"Ollama embeddings HTTP {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        embedding = data.get("embedding")
        if not embedding:
            raise LLMProviderError("Ollama embeddings returned no embedding")
        return [float(x) for x in embedding]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_one(query)

    def dimension(self) -> int:
        return self._dim or 768

    @property
    def model(self) -> str:
        return self._model
