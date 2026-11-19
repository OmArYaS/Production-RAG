"""Health check router."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    container = request.app.state.container
    return {
        "status": "ok",
        "embedding_backend": getattr(container.embedding, "backend", "ollama"),
        "embedding_model": getattr(container.embedding, "model_name", None),
        "llm_model": container.llm.model,
        "vector_index": type(container.vector_index).__name__,
        "document_count": len(container.document_repo.list_documents()),
    }
