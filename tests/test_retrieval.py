"""Tests for hybrid retrieval (dense + sparse, RRF fused)."""

import pytest

from application.retrieval import HybridRetrieveUseCase
from domain.exceptions import EmbeddingDimensionError
from domain.models import ChunkRecord, DocumentRecord
from tests.fakes import (
    FakeDocumentRepository,
    FakeEmbedding,
    FakeKeywordIndex,
    FakeVectorIndex,
)


def _make_retrieval(vector_results, keyword_results, dimension=8, doc_count=1):
    embedding = FakeEmbedding(dim=8)
    doc_repo = FakeDocumentRepository()
    for i in range(doc_count):
        doc_repo.save_document(
            DocumentRecord(doc_id=f"doc{i}", source_name="s.txt", content_type="text/plain")
        )
    # Put chunks into the store so get_chunk works.
    for cid, doc_id, text in [
        ("c1", "doc0", "chunk one text"),
        ("c2", "doc0", "chunk two text"),
        ("c3", "doc0", "chunk three text"),
    ]:
        doc_repo.save_chunk(
            ChunkRecord(chunk_id=cid, doc_id=doc_id, source_name="s.txt", chunk_index=0, text=text)
        )
    vector = FakeVectorIndex(dimension=dimension, search_results=vector_results)
    keyword = FakeKeywordIndex(search_results=keyword_results)
    retrieve = HybridRetrieveUseCase(vector, keyword, embedding, doc_repo, top_k=10)
    return retrieve


def test_hybrid_fusion_union():
    retrieve = _make_retrieval(
        vector_results=[("c1", 0.9), ("c2", 0.8)],
        keyword_results=[("c2", 0.7), ("c3", 0.6)],
    )
    hits = retrieve.execute("query")
    ids = [h.chunk_id for h in hits]
    assert set(ids) == {"c1", "c2", "c3"}
    # c2 has best RRF (rank1 dense + rank1 sparse).
    assert ids[0] == "c2"
    by_id = {h.chunk_id: h for h in hits}
    assert by_id["c1"].source_name == "s.txt"
    assert by_id["c2"].text == "chunk two text"
    assert all(h.score > 0 for h in hits)


def test_empty_knowledge_base_returns_empty():
    retrieve = _make_retrieval(vector_results=[("c1", 0.9)], keyword_results=[("c1", 0.9)], doc_count=0)
    assert retrieve.execute("anything") == []


def test_zero_hits_query():
    retrieve = _make_retrieval(vector_results=[], keyword_results=[], doc_count=1)
    assert retrieve.execute("nomatch") == []


def test_dimension_mismatch_raises():
    # Query vector has dim 8 (FakeEmbedding), but the vector index expects 7.
    retrieve = _make_retrieval(vector_results=[("c1", 0.9)], keyword_results=[("c1", 0.9)], dimension=7, doc_count=1)
    with pytest.raises(EmbeddingDimensionError):
        retrieve.execute("query")
