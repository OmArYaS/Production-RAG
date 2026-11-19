"""Q&A history router."""

from __future__ import annotations

from fastapi import APIRouter, Request

from domain.models import QueryLogEntry

router = APIRouter(tags=["history"])


@router.get("/api/history", response_model=list[QueryLogEntry])
async def get_history(request: Request, limit: int = 100):
    container = request.app.state.container
    return container.list_history.execute(limit=limit)
