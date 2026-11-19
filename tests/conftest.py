"""Shared pytest fixtures (offline, deterministic)."""

from __future__ import annotations

import os
import tempfile

import pytest

from application.deletion import DeleteDocumentUseCase
from application.history import ListHistoryUseCase
from application.ingestion import IngestDocumentUseCase
from application.retrieval import HybridRetrieveUseCase
from application.synthesis import SynthesizeAnswerUseCase
from infrastructure.config import Settings
from infrastructure.di import Container, build_container
from presentation.main import create_app

from tests.fakes import (
    FakeDocumentRepository,
    FakeEmbedding,
    FakeKeywordIndex,
    FakeLLM,
    FakeQueryLogRepository,
    FakeVectorIndex,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def settings(db_path) -> Settings:
    return Settings(
        DATABASE_URL=f"sqlite:///{db_path}",
        EMBEDDING_PROVIDER="local",
    )


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding(dim=8)


@pytest.fixture
def container(fake_embedding, db_path) -> Container:
    """A fully-faked container (all adapters are deterministic fakes)."""
    s = Settings(DATABASE_URL=f"sqlite:///{db_path}")
    doc_repo = FakeDocumentRepository()
    vector = FakeVectorIndex(dimension=8)
    keyword = FakeKeywordIndex()
    llm = FakeLLM()
    qlog = FakeQueryLogRepository()
    return Container(
        settings=s,
        embedding=fake_embedding,
        vector_index=vector,
        keyword_index=keyword,
        llm=llm,
        document_repo=doc_repo,
        query_log_repo=qlog,
        ingest=IngestDocumentUseCase(doc_repo, vector, keyword, fake_embedding),
        retrieve=HybridRetrieveUseCase(vector, keyword, fake_embedding, doc_repo),
        synthesize=SynthesizeAnswerUseCase(llm),
        delete_document=DeleteDocumentUseCase(doc_repo, vector, keyword),
        list_history=ListHistoryUseCase(qlog),
    )


@pytest.fixture
def client(container):
    from fastapi.testclient import TestClient

    app = create_app(container)
    return TestClient(app)


@pytest.fixture
def real_container(settings) -> Container:
    """Container with real local adapters (offline fallbacks) + FakeLLM."""
    c = build_container(settings)
    c.llm = FakeLLM()
    c.synthesize = SynthesizeAnswerUseCase(llm_provider=c.llm)
    return c


@pytest.fixture
def real_client(real_container):
    from fastapi.testclient import TestClient

    app = create_app(real_container)
    return TestClient(app)


def sample_txt_bytes() -> bytes:
    return (
        "Project Zephyr: A modular RAG assistant.\n\n"
        "## Architecture\n\n"
        "The system uses clean architecture with four layers.\n"
        "Retrieval is hybrid: dense vector search plus sparse BM25.\n\n"
        "## Operations\n\n"
        "To deploy, run docker-compose up -d. The API exposes /api/documents.\n"
    ).encode()
