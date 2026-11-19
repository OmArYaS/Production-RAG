"""Hybrid retrieval use case: dense + sparse fused via Reciprocal Rank Fusion.

Depends ONLY on ``domain/`` ports. Returns :class:`SearchHit` objects.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from domain.exceptions import EmbeddingDimensionError
from domain.models import SearchHit
from domain.ports import (
    DocumentRepository,
    EmbeddingProvider,
    KeywordIndex,
    VectorIndex,
)

from application.rrf import rrf_fuse


class HybridRetrieveUseCase:
    """Retrieve the most relevant chunks using dense semantic search +
    BM25 lexical search, fused with Reciprocal Rank Fusion.

    Gracefully handles an empty knowledge base (returns []), chunks that appear
    in only one of the two indexes, and zero-hit queries.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        embedding_provider: EmbeddingProvider,
        document_repo: DocumentRepository,
        top_k: int = 10,
    ) -> None:
        self._vector_index = vector_index
        self._keyword_index = keyword_index
        self._embedding = embedding_provider
        self._doc_repo = document_repo
        self._top_k = top_k

    def execute(self, query: str) -> List[SearchHit]:
        if not query or not query.strip():
            return []

        # If nothing is indexed yet, skip search entirely (no crash).
        if not self._doc_repo.list_documents():
            return []

        # Dimension safety: query vector must match the index dimension.
        query_vector = self._embedding.embed_query(query)
        expected_dim = self._vector_index.dimension()
        if len(query_vector) != expected_dim:
            raise EmbeddingDimensionError(
                expected=expected_dim, actual=len(query_vector)
            )

        # Dense semantic search.
        dense_hits = self._vector_index.search(query_vector, self._top_k)
        # Sparse BM25 lexical search.
        sparse_hits = self._keyword_index.search(query, self._top_k)

        # Fuse via RRF. Handles dense-only, sparse-only, and zero-hit queries.
        fused = rrf_fuse(dense_hits, sparse_hits, self._top_k)

        hits: List[SearchHit] = []
        for chunk_id, score in fused:
            chunk = self._doc_repo.get_chunk(chunk_id)
            if chunk is None:
                # Chunk was removed from the index but not the store (or vice
                # versa); skip rather than crash.
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_name=chunk.source_name,
                    text=chunk.text,
                    score=score,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    chunk_index=chunk.chunk_index,
                )
            )
        return hits
