"""Grounded answer synthesis with strict citations and abstention.

Depends ONLY on ``domain/`` ports and pure helpers. Instructs the LLM to answer
from retrieved context only, cite inline as [N], and abstain when evidence is
insufficient. Malformed/empty LLM output is handled gracefully.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple

from domain.exceptions import InsufficientContextError, LLMProviderError
from domain.models import AnswerResponse, Citation, SearchHit
from domain.ports import LLMProvider

# Marker the LLM is instructed to emit when it cannot answer from context.
_ABSTAIN_MARKER = "ABSTAIN"


def strip_llm_output(text: Optional[str]) -> str:
    """Strip reasoning tags and markdown fences from raw LLM output.

    Removes ``<think>...</think>`` reasoning blocks and ``` fence markers,
    keeping the visible answer text, then trims whitespace. Never crashes on
    ``None`` or malformed input.
    """
    if not text:
        return ""
    cleaned = text
    # Remove <think>...</think> blocks (non-greedy, case-insensitive, multiline).
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # Remove the opening fence line (```lang) keeping content.
    cleaned = re.sub(r"```[^\n`]*\n?", "", cleaned)
    # Remove any leftover fence markers.
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def _extract_cited_indices(answer: str, max_index: int) -> List[int]:
    """Return citation indices referenced in the answer, clamped to range."""
    seen: List[int] = []
    for m in re.finditer(r"\[(\d+)\]", answer):
        idx = int(m.group(1))
        if 1 <= idx <= max_index and idx not in seen:
            seen.append(idx)
    return seen


class SynthesizeAnswerUseCase:
    """Synthesize a grounded answer from retrieved chunks."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_context_chunks: int = 8,
        max_snippet_chars: int = 400,
    ) -> None:
        self._llm = llm_provider
        self._max_context_chunks = max_context_chunks
        self._max_snippet_chars = max_snippet_chars

    def execute(self, question: str, hits: List[SearchHit]) -> AnswerResponse:
        start = time.perf_counter()

        # No retrieved context: abstain without calling the LLM.
        if not hits:
            latency_ms = (time.perf_counter() - start) * 1000
            return AnswerResponse(
                answer=(
                    "No indexed documents contain information relevant to your "
                    "question. Please upload documents first."
                ),
                citations=[],
                abstained=True,
                model=self._llm.model,
                latency_ms=latency_ms,
                hits_count=0,
            )

        context_text, index_to_hit = self._build_context(hits)
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(question, context_text)

        try:
            raw = self._llm.generate(system_prompt, user_prompt)
        except Exception as exc:  # unreachable Ollama / provider failure
            raise LLMProviderError(str(exc))

        answer = strip_llm_output(raw)
        if not answer:
            # Malformed/empty LLM response: treat as abstention.
            latency_ms = (time.perf_counter() - start) * 1000
            return AnswerResponse(
                answer=(
                    "The language model returned an empty response. The "
                    "information could not be confirmed from the provided "
                    "documents."
                ),
                citations=[],
                abstained=True,
                model=self._llm.model,
                latency_ms=latency_ms,
                hits_count=len(hits),
            )

        abstained = answer.strip().upper().startswith(_ABSTAIN_MARKER)
        if abstained:
            # Use the LLM's own abstention statement when present, else a clear default.
            body = answer.strip()
            if body.upper().startswith(_ABSTAIN_MARKER):
                body = body[len(_ABSTAIN_MARKER):].strip(" :\t-")
            if not body:
                body = (
                    "The information requested is not found in the provided "
                    "documents."
                )
            latency_ms = (time.perf_counter() - start) * 1000
            return AnswerResponse(
                answer=body,
                citations=[],
                abstained=True,
                model=self._llm.model,
                latency_ms=latency_ms,
                hits_count=len(hits),
            )

        # Build citations only for indices actually cited and in range.
        cited_indices = _extract_cited_indices(answer, len(hits))
        citations: List[Citation] = []
        for idx in cited_indices:
            hit = index_to_hit[idx]
            snippet = hit.text[: self._max_snippet_chars]
            if len(hit.text) > self._max_snippet_chars:
                snippet += "..."
            citations.append(
                Citation(
                    index=idx,
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    source_name=hit.source_name,
                    page_number=hit.page_number,
                    section_heading=hit.section_heading,
                    chunk_index=hit.chunk_index,
                    snippet=snippet,
                    score=hit.score,
                )
            )

        latency_ms = (time.perf_counter() - start) * 1000
        return AnswerResponse(
            answer=answer,
            citations=citations,
            abstained=False,
            model=self._llm.model,
            latency_ms=latency_ms,
            hits_count=len(hits),
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a grounded Q&A assistant. Answer ONLY using the provided "
            "context chunks. Do not use outside knowledge or hallucinate.\n"
            "Each context chunk is labeled with a citation number [1], [2], ... "
            "along with its source_name, page_number, and chunk_id.\n"
            "Rules:\n"
            "1. Cite every factual claim with inline citations like [1], [2].\n"
            "2. Use only citation numbers that refer to the provided context.\n"
            "3. If the answer cannot be determined from the context, reply with "
            "exactly: ABSTAIN\n"
            "4. Be concise and accurate.\n"
        )

    @staticmethod
    def _user_prompt(question: str, context_text: str) -> str:
        return (
            "Context:\n"
            + context_text
            + "\n\nQuestion: "
            + question
            + "\n\nAnswer:"
        )

    def _build_context(self, hits: List[SearchHit]) -> Tuple[str, dict]:
        """Build numbered context text and a 1-based index -> hit map."""
        lines: List[str] = []
        index_to_hit: dict = {}
        for i, hit in enumerate(hits[: self._max_context_chunks], start=1):
            index_to_hit[i] = hit
            page = f", page={hit.page_number}" if hit.page_number is not None else ""
            section = f", section={hit.section_heading}" if hit.section_heading else ""
            lines.append(
                f"[{i}] source={hit.source_name}{page}{section}, chunk_id={hit.chunk_id}\n"
                f"{hit.text}"
            )
        return "\n\n".join(lines), index_to_hit
