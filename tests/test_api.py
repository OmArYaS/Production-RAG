"""API endpoint tests (using fully faked adapters)."""

import pytest
from tests.conftest import sample_txt_bytes


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_documents_empty_initially(client):
    r = client.get("/api/documents")
    assert r.status_code == 200
    assert r.json() == []


def test_query_blank_question_422(client):
    assert client.post("/api/query", json={"question": "   "}).status_code == 422
    assert client.post("/api/query", json={"question": ""}).status_code == 422


def test_query_no_documents_abstains(client):
    r = client.post("/api/query", json={"question": "hello"})
    assert r.status_code == 200
    assert r.json()["abstained"] is True


def test_delete_missing_404(client):
    assert client.delete("/api/documents/does-not-exist").status_code == 404


def test_upload_unsupported_format_400(client):
    r = client.post(
        "/api/documents", files={"file": ("virus.exe", b"MZ...", "application/octet-stream")}
    )
    assert r.status_code == 400


def test_upload_and_list_and_delete(client):
    r = client.post("/api/documents", files={"file": ("a.txt", sample_txt_bytes(), "text/plain")})
    assert r.status_code == 201
    doc = r.json()
    assert doc["source_name"] == "a.txt"
    assert doc["doc_id"]
    assert doc["chunk_count"] >= 1

    # List includes the doc.
    r2 = client.get("/api/documents")
    assert r2.status_code == 200
    assert len(r2.json()) == 1

    # Delete removes it.
    r3 = client.delete(f"/api/documents/{doc['doc_id']}")
    assert r3.status_code == 200
    assert client.get("/api/documents").json() == []


def test_history_endpoint(client):
    # Query with no docs -> persists an abstained log entry.
    client.post("/api/query", json={"question": "hi?"})
    r = client.get("/api/history")
    assert r.status_code == 200
    assert len(r.json()) >= 1
    entry = r.json()[0]
    assert entry["question"] == "hi?"
    assert "answer" in entry
