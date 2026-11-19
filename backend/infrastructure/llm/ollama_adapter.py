"""Ollama chat LLM adapter (model-agnostic local chat LLM).

Default model: ``qwen2.5:7b-instruct-q4_k_m`` (swappable via OLLAMA_MODEL;
gemma2, deepseek-r1, llama3.1, etc. require no code changes). This adapter is
NEVER used for embeddings — a dedicated embedding provider is separate.
"""

from __future__ import annotations

from typing import Optional

import requests

from application.synthesis import strip_llm_output
from domain.exceptions import LLMProviderError
from domain.ports import LLMProvider


class OllamaLLMAdapter(LLMProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct-q4_k_m",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise LLMProviderError(f"Ollama unreachable at {self._base_url}: {exc}")

        if resp.status_code >= 400:
            raise LLMProviderError(
                f"Ollama HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMProviderError("Ollama returned invalid JSON")

        message = data.get("message", {}) or {}
        content: Optional[str] = message.get("content")
        if not content:
            # Some models return empty content when they emit reasoning only.
            if data.get("done") and not content:
                return ""
            raise LLMProviderError("Ollama returned an empty chat message")

        # Strip reasoning tags and markdown fences.
        return strip_llm_output(content)
