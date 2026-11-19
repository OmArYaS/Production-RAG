"""Dependency injection container — the composition root.

Wires concrete infrastructure adapters to application use cases. Swapping any
adapter (embedding, vector, keyword, LLM) only requires changing it here and
an adapter — never domain/application code.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.deletion import DeleteDocumentUseCase
from application.history import ListHistoryUseCase
from application.ingestion import IngestDocumentUseCase
from application.retrieval import HybridRetrieveUseCase
from application.synthesis import SynthesizeAnswerUseCase
from domain.ports import (
    DocumentRepository,
    EmbeddingProvider,
    KeywordIndex,
    LLMProvider,
    QueryLogRepository,
    VectorIndex,
)
from infrastructure.config import Settings
from infrastructure.embedding import build_embedding_provider
from infrastructure.keyword import build_keyword_index
from infrastructure.llm import build_llm_provider
from infrastructure.repositories import build_sqlite_repos
from infrastructure.vector import build_vector_index


@dataclass
class Container:
    settings: Settings
    embedding: EmbeddingProvider
    vector_index: VectorIndex
    keyword_index: KeywordIndex
    llm: LLMProvider
    document_repo: DocumentRepository
    query_log_repo: QueryLogRepository
    ingest: IngestDocumentUseCase
    retrieve: HybridRetrieveUseCase
    synthesize: SynthesizeAnswerUseCase
    delete_document: DeleteDocumentUseCase
    list_history: ListHistoryUseCase


def build_container(settings: Settings) -> Container:
    """Build the full wired container from settings."""
    embedding = build_embedding_provider(settings)
    vector_index = build_vector_index(settings, embedding.dimension())
    keyword_index = build_keyword_index()
    llm = build_llm_provider(settings)
    document_repo, query_log_repo = build_sqlite_repos(settings.DATABASE_URL)

    ingest = IngestDocumentUseCase(
        document_repo=document_repo,
        vector_index=vector_index,
        keyword_index=keyword_index,
        embedding_provider=embedding,
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP,
    )
    retrieve = HybridRetrieveUseCase(
        vector_index=vector_index,
        keyword_index=keyword_index,
        embedding_provider=embedding,
        document_repo=document_repo,
        top_k=settings.TOP_K,
    )
    synthesize = SynthesizeAnswerUseCase(llm_provider=llm)
    delete_document = DeleteDocumentUseCase(
        document_repo=document_repo,
        vector_index=vector_index,
        keyword_index=keyword_index,
    )
    list_history = ListHistoryUseCase(query_log_repo)

    return Container(
        settings=settings,
        embedding=embedding,
        vector_index=vector_index,
        keyword_index=keyword_index,
        llm=llm,
        document_repo=document_repo,
        query_log_repo=query_log_repo,
        ingest=ingest,
        retrieve=retrieve,
        synthesize=synthesize,
        delete_document=delete_document,
        list_history=list_history,
    )
