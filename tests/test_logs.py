import json
from pathlib import Path

from fastapi.testclient import TestClient

import app.indexing as indexing
import app.main as main


def test_logs_include_upload_and_query(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_DISABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "4")
    monkeypatch.setenv("CHUNK_SIZE", "12")
    monkeypatch.setenv("CHUNK_OVERLAP", "2")
    monkeypatch.setattr(main, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(main, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(main, "LOGS_ROOT", tmp_path / "logs")
    monkeypatch.setattr(indexing, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(indexing, "INDEX_ROOT", tmp_path / "index")

    client = TestClient(main.app)
    upload_response = client.post(
        "/upload",
        files={"file": ("note.txt", b"alpha beta gamma", "text/plain")},
    )
    assert upload_response.status_code == 200
    payload = upload_response.json()
    doc_id = payload["doc_id"]

    chunks_path = main.CHUNKS_ROOT / doc_id / "chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]

    query_response = client.post(
        "/query",
        json={"doc_id": doc_id, "question": chunks[0]["text"], "top_k": 1},
    )
    assert query_response.status_code == 200

    logs_response = client.get(f"/logs?n=10&doc_id={doc_id}")
    assert logs_response.status_code == 200
    events = logs_response.json()["events"]
    event_types = {event["event_type"] for event in events}
    assert "upload" in event_types
    assert "query" in event_types
