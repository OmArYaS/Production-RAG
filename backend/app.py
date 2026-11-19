"""Uvicorn entrypoint for the FastAPI app."""

from presentation.main import create_app

app = create_app()
