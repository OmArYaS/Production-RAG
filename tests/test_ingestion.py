"""Tests for ingestion, parsing, and their edge cases."""

import pytest

from application.ingestion import IngestDocumentUseCase
from domain.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    UnsupportedFileFormatError,
)
from domain.models import ChunkRecord, DocumentRecord
from domain.ports import DocumentRepository, EmbeddingProvider, KeywordIndex, VectorIndex
from infrastructure.parsing import get_content_type, is_supported, parse_document

from tests.conftest import sample_txt_bytes


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------
def test_parser_txt():
    tb = parse_document("a.txt", b"hello world")
    assert tb == [(None, "hello world")]
    assert is_supported("a.txt")


def test_parser_md():
    tb = parse_document("a.md", b"# Title\n\ntext")
    assert tb == [(None, "# Title\n\ntext")]
    assert get_content_type("a.md") == "text/markdown"


def test_parser_unsupported():
    for fn in ["a.exe", "a.zip", "a.docx", "noext"]:
        assert not is_supported(fn)
        with pytest.raises(UnsupportedFileFormatError):
            parse_document(fn, b"data")


def test_parser_corrupted_pdf():
    with pytest.raises(Exception):
        parse_document("a.pdf", b"this is not a real pdf")


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------
def test_ingest_txt_returns_document_and_chunks(container):
    data = sample_txt_bytes()
    doc = container.ingest.execute("sample.txt", [(None, data.decode())], "text/plain", len(data))
    assert isinstance(doc, DocumentRecord)
    assert doc.source_name == "sample.txt"
    assert doc.chunk_count >= 1
    assert container.document_repo.get_document(doc.doc_id) is not None
    chunks = container.document_repo.get_chunks_by_document(doc.doc_id)
    assert len(chunks) == doc.chunk_count
    assert all(c.page_number is None for c in chunks)
    # vector + keyword indexes received chunks
    assert len(container.vector_index._vectors) == doc.chunk_count
    assert len(container.keyword_index._chunks) == doc.chunk_count


def test_ingest_empty_file_rejected(container):
    with pytest.raises(EmptyDocumentError):
        container.ingest.execute("empty.txt", [(None, "")], "text/plain", 0)


def test_ingest_whitespace_only_rejected(container):
    with pytest.raises(EmptyDocumentError):
        container.ingest.execute("ws.txt", [(None, "   \n\t  ")], "text/plain", 10)


def test_duplicate_upload_replaces(container):
    doc1 = container.ingest.execute("dup.txt", [(None, "some content here.")], "text/plain", 1)
    doc2 = container.ingest.execute("dup.txt", [(None, "other content here.")], "text/plain", 2)
    # Same doc_id kept, no duplicate documents or chunks.
    assert doc1.doc_id == doc2.doc_id
    docs = [d for d in container.document_repo.list_documents() if d.source_name == "dup.txt"]
    assert len(docs) == 1
    chunks = container.document_repo.get_chunks_by_document(doc1.doc_id)
    assert all(c.text != "some content here." for c in chunks)
    assert len(container.vector_index._vectors) == len(chunks)
    # No orphan chunks in the fake keyword index.
    orphan = [cid for cid, c in container.keyword_index._chunks.items() if c.doc_id != doc1.doc_id and c.source_name == "dup.txt"]
    assert orphan == []
