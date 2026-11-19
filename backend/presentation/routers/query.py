"""Q&A router: ask grounded questions."""

from __future__ import annotations

from fastapi import APIRouter, Request

from domain.models import AnswerResponse, QueryLogEntry
from presentation.schemas import QueryRequest

router = APIRouter(tags=["qa"])


def _container(request: Request):
    return request.app.state.container


@router.post("/api/query", response_model=AnswerResponse)
async def ask_question(body: QueryRequest, request: Request) -> AnswerResponse:
    container = _container(request)
    question = body.question.strip()

    # Hybrid retrieval (dense + BM25, RRF fused). Returns [] when KB is empty.
    hits = container.retrieve.execute(question)

    # Grounded synthesis with citations + abstention.
    response = container.synthesize.execute(question, hits)

    # Persist Q&A history.
    entry = QueryLogEntry(
        question=question,
        answer=response.answer,
        citations=response.citations,
        abstained=response.abstained,
        model=response.model,
        latency_ms=response.latency_ms,
        hits_count=response.hits_count,
    )
    container.query_log_repo.save_entry(entry)
    return response
