"""Tests for smart chunking and metadata preservation."""

import pytest

from application.chunking import chunk_document


def _text_with_headings() -> str:
    return (
        "Intro paragraph about the system.\n\n"
        "## Architecture\n\n"
        "The system uses clean architecture with four layers.\n\n"
        "Retrieval is hybrid: dense vector search plus sparse BM25.\n\n"
        "## Operations\n\n"
        "To deploy, run docker-compose up -d. The API exposes /api/documents.\n"
    )


def test_chunks_have_metadata():
    text = _text_with_headings()
    chunks = chunk_document([(1, text)], source_name="doc.pdf", doc_id="doc1")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.doc_id == "doc1"
        assert c.source_name == "doc.pdf"
        assert c.page_number == 1
        assert isinstance(c.chunk_index, int)
    # Distinct chunk_index values
    assert len({c.chunk_index for c in chunks}) == len(chunks)


def test_section_heading_detected():
    text = _text_with_headings()
    chunks = chunk_document([(1, text)], source_name="doc.pdf", doc_id="doc1")
    headings = [c.section_heading for c in chunks if c.section_heading]
    assert any("Architecture" in h for h in headings)
    assert any("Operations" in h for h in headings)


def test_txt_page_number_none():
    text = "Some plain text about retrieval. " * 10
    chunks = chunk_document([(None, text)], source_name="a.txt", doc_id="d1")
    assert chunks
    assert all(c.page_number is None for c in chunks)


def test_chunk_size_respects_target():
    # Long text with small chunk_size -> multiple chunks each <= chunk_size words.
    text = ("word " * 1000).strip()
    chunks = chunk_document([(1, text)], source_name="d", doc_id="d", chunk_size=20, overlap=5)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 20


def test_overlap_creates_shared_words():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_document([(1, text)], source_name="d", doc_id="d", chunk_size=30, overlap=5)
    assert len(chunks) > 1
    # With overlap, some words repeat across consecutive chunks.
    c0 = chunks[0].text.split()
    c1 = chunks[1].text.split()
    assert len(set(c0) & set(c1)) > 0


def test_empty_and_whitespace_texts_return_empty():
    assert chunk_document([(1, "")], "d", "d") == []
    assert chunk_document([(1, "   \n  ")], "d", "d") == []
    assert chunk_document([], "d", "d") == []


def test_invalid_chunk_params():
    with pytest.raises(ValueError):
        chunk_document([(1, "hello")], "d", "d", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_document([(1, "hello")], "d", "d", overlap=-1)
    with pytest.raises(ValueError):
        chunk_document([(1, "hello")], "d", "d", overlap=10, chunk_size=5)
