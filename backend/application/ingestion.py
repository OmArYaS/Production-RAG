"""Application use cases for the Hybrid Document & Knowledge Assistant.

Each use case depends ONLY on ``domain/`` ports and pure application helpers.
Infrastructure adapters are injected at construction time (DI), so swapping an
adapter never touches this layer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from domain.exceptions import DomainError, EmptyDocumentError, EmbeddingDimensionError
from domain.models import ChunkRecord, DocumentRecord
from domain.ports import (
    DocumentRepository,
    EmbeddingProvider,
    KeywordIndex,
    VectorIndex,
)

from application.chunking import chunk_document


class IngestDocumentUseCase:
    """Ingest a document: parse -> chunk -> embed -> store + index.

    Re-uploading a document with the same ``source_name`` deterministically
    replaces the previous version (document, chunks, vector and keyword index
    entries) without leaving orphans or duplicate drift.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        embedding_provider: EmbeddingProvider,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        self._doc_repo = document_repo
        self._vector_index = vector_index
        self._keyword_index = keyword_index
        self._embedding = embedding_provider
        self._chunk_size = chunk_size
        self._overlap = overlap

    def execute(
        self,
        source_name: str,
        text_by_page: List[Tuple[Optional[int], str]],
        content_type: str,
        size_bytes: int = 0,
        file_path: Optional[str] = None,
    ) -> DocumentRecord:
        # Reject documents with zero extractable text before touching indexes.
        full_text = "\n".join(text for _, text in text_by_page if text)
        if not full_text or not full_text.strip():
            raise EmptyDocumentError(
                source_name,
                reason=(
                    "no extractable text (empty, whitespace-only, or scanned/image-only)"
                ),
            )

        # Deterministic replace: remove any existing document with the same
        # source name, including its chunks from all indexes.
        existing = self._find_existing_by_source(source_name)
        if existing is not None:
            self._remove_document_completely(existing.doc_id)

        # Smart chunking with rich metadata.
        chunks = chunk_document(
            text_by_page,
            source_name=source_name,
            doc_id="",  # filled after the document is saved
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )
        if not chunks:
            raise EmptyDocumentError(source_name, reason="chunking produced no chunks")

        # Persist document record first so chunks have a doc_id.
        document = DocumentRecord(
            source_name=source_name,
            file_path=file_path,
            content_type=content_type,
            size_bytes=size_bytes,
            chunk_count=len(chunks),
            status="indexed",
        )
        if existing is not None:
            document.doc_id = existing.doc_id  # keep a stable doc_id on replace
        self._doc_repo.save_document(document)

        # Assign doc_id to chunks and persist chunk metadata.
        for chunk in chunks:
            chunk.doc_id = document.doc_id
            self._doc_repo.save_chunk(chunk)

        # Embed chunk texts (dedicated embedding model, never the chat LLM).
        chunk_texts = [c.text for c in chunks]
        vectors = self._embedding.embed_texts(chunk_texts)
        if len(vectors) != len(chunks):
            raise DomainError(
                f"Embedding count mismatch: {len(chunks)} chunks but "
                f"{len(vectors)} vectors produced.",
                status_code=500,
            )

        # Dimension safety: verify the embedding dimension matches the index.
        expected_dim = self._vector_index.dimension()
        actual_dim = self._embedding.dimension()
        if expected_dim != actual_dim:
            raise EmbeddingDimensionError(expected=expected_dim, actual=actual_dim)
        for v in vectors:
            if len(v) != expected_dim:
                raise EmbeddingDimensionError(expected=expected_dim, actual=len(v))

        # Index dense vectors and sparse keyword entries.
        items = [(c.chunk_id, c.doc_id, v) for c, v in zip(chunks, vectors)]
        self._vector_index.add(items)
        self._keyword_index.add(chunks)

        document.chunk_count = len(chunks)
        self._doc_repo.save_document(document)
        return document

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_existing_by_source(self, source_name: str) -> Optional[DocumentRecord]:
        for doc in self._doc_repo.list_documents():
            if doc.source_name == source_name and doc.status != "deleted":
                return doc
        return None

    def _remove_document_completely(self, doc_id: str) -> None:
        # Remove chunks from keyword + vector indexes.
        self._keyword_index.delete(doc_id)
        self._vector_index.delete(doc_id)
        # Remove chunk metadata.
        self._doc_repo.delete_chunks_by_document(doc_id)
        # Remove document record.
        self._doc_repo.delete_document(doc_id)
