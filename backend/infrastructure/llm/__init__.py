"""LLM provider adapters (model-agnostic chat LLM)."""

from domain.ports import LLMProvider

from .ollama_adapter import OllamaLLMAdapter


def build_llm_provider(settings) -> LLMProvider:
    """Factory: build the configured local chat LLM provider (Ollama)."""
    return OllamaLLMAdapter(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
    )
