"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from infrastructure.config import Settings
from infrastructure.di import Container, build_container
from presentation.error_handlers import register_exception_handlers
from presentation.routers import documents, health, history, query


def create_app(container: Container | None = None) -> FastAPI:
    settings = Settings.from_env()
    if container is None:
        container = build_container(settings)

    app = FastAPI(
        title="Hybrid Document & Knowledge Assistant",
        description="Local RAG assistant: hybrid dense+BM25 retrieval, "
        "grounded synthesis with citations, and full persistence.",
        version="1.0.0",
    )
    app.state.container = container

    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(history.router)
    app.include_router(health.router)
    register_exception_handlers(app)
    return app
