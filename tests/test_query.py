from pathlib import Path

from fastapi.testclient import TestClient

import app.indexing as indexing
import app.main as main


def test_query_returns_stable_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_DISABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "4")
    monkeypatch.setenv("CHUNK_SIZE", "12")
    monkeypatch.setenv("CHUNK_OVERLAP", "2")
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(main, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(indexing, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(indexing, "INDEX_ROOT", tmp_path / "index")

    doc_id = "doc-query"
    processed_dir = main.PROCESSED_ROOT / doc_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "text.txt").write_text("alpha beta gamma delta epsilon", encoding="utf-8")

    chunks = main.build_chunks(doc_id)
    indexing.index_doc(doc_id)

    client = TestClient(main.app)
    response = client.post(
        "/query",
        json={"doc_id": doc_id, "question": chunks[0]["text"], "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_id"] == doc_id
    assert payload["top_k"] == 1
    assert payload["results"][0]["chunk_id"] == chunks[0]["chunk_id"]


def test_query_returns_insufficient_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_DISABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "4")
    monkeypatch.setenv("CHUNK_SIZE", "12")
    monkeypatch.setenv("CHUNK_OVERLAP", "2")
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.5")
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(main, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(indexing, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(indexing, "INDEX_ROOT", tmp_path / "index")

    doc_id = "doc-insufficient"
    processed_dir = main.PROCESSED_ROOT / doc_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "text.txt").write_text("alpha beta gamma", encoding="utf-8")

    chunks = main.build_chunks(doc_id)
    indexing.index_doc(doc_id)

    client = TestClient(main.app)
    response = client.post(
        "/query",
        json={"doc_id": doc_id, "question": chunks[0]["text"], "top_k": 1},
    )

    payload = response.json()
    assert payload["answer"] == "Insufficient evidence in provided documents."
    assert payload["citations"] == []


def test_query_returns_citations_with_answer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_DISABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "4")
    monkeypatch.setenv("CHUNK_SIZE", "12")
    monkeypatch.setenv("CHUNK_OVERLAP", "2")
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "-1.0")
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(main, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(indexing, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(indexing, "INDEX_ROOT", tmp_path / "index")

    doc_id = "doc-answer"
    processed_dir = main.PROCESSED_ROOT / doc_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "text.txt").write_text("delta epsilon zeta", encoding="utf-8")

    chunks = main.build_chunks(doc_id)
    indexing.index_doc(doc_id)

    client = TestClient(main.app)
    response = client.post(
        "/query",
        json={"doc_id": doc_id, "question": chunks[0]["text"], "top_k": 1},
    )

    payload = response.json()
    assert payload["answer"]
    assert payload["citations"]
    assert payload["citations"][0]["chunk_id"] == chunks[0]["chunk_id"]
