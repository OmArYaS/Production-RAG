"""End-to-end API integration tests using real (offline) adapters + FakeLLM."""

from tests.conftest import sample_txt_bytes


def test_upload_query_history_delete_flow(real_client, real_container):
    # Upload
    r = real_client.post("/api/documents", files={"file": ("sample.txt", sample_txt_bytes(), "text/plain")})
    assert r.status_code == 201
    doc = r.json()
    assert doc["source_name"] == "sample.txt"
    assert doc["chunk_count"] >= 1

    # Documents list
    assert real_client.get("/api/documents").json()[0]["doc_id"] == doc["doc_id"]

    # Ask a question (FakeLLM returns canned text)
    qr = real_client.post("/api/query", json={"question": "What is the architecture?"})
    assert qr.status_code == 200
    assert qr.json()["model"] == "fake-model"
    assert qr.json()["hits_count"] >= 0
    assert isinstance(qr.json()["answer"], str) and qr.json()["answer"]

    # History persisted
    hist = real_client.get("/api/history").json()
    assert len(hist) >= 1
    assert hist[0]["question"] == "What is the architecture?"

    # Delete removes doc + chunks from indexes
    dr = real_client.delete(f"/api/documents/{doc['doc_id']}")
    assert dr.status_code == 200
    assert real_client.get("/api/documents").json() == []
    assert len(real_container.vector_index._vectors) == 0
    assert len(real_container.keyword_index._chunks) == 0


def test_reupload_replaces_document(real_client, real_container):
    r1 = real_client.post("/api/documents", files={"file": ("rep.txt", b"first version text", "text/plain")})
    assert r1.status_code == 201
    doc_id = r1.json()["doc_id"]

    r2 = real_client.post("/api/documents", files={"file": ("rep.txt", b"second version text", "text/plain")})
    assert r2.status_code == 201
    assert r2.json()["doc_id"] == doc_id
    docs = real_client.get("/api/documents").json()
    assert len([d for d in docs if d["source_name"] == "rep.txt"]) == 1
