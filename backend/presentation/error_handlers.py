"""HTTP error mapping from domain exceptions to FastAPI responses."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from domain.exceptions import DocumentNotFoundError, DomainError


def map_domain_error(exc: DomainError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _handle_domain_error(request, exc: DomainError):  # noqa: ARG001
        return map_domain_error(exc)

    @app.exception_handler(DocumentNotFoundError)
    async def _handle_not_found(request, exc: DocumentNotFoundError):  # noqa: ARG001
        return JSONResponse(
            status_code=404,
            content={"detail": exc.message},
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request, exc: Exception):  # noqa: ARG001
        from fastapi.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code, content={"detail": exc.detail}
            )
        # Bubble unexpected errors as 500s; never a silent crash.
        return JSONResponse(status_code=500, content={"detail": str(exc)})
