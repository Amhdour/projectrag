import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main


def test_upload_file_stores_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
    monkeypatch.setattr(main, "UPLOAD_ROOT", tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/upload",
        files={"file": ("sample.txt", io.BytesIO(b"hello"), "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample.txt"
    assert payload["status"] == "stored"

    doc_path = tmp_path / payload["doc_id"] / "sample.txt"
    assert doc_path.exists()
    metadata_path = tmp_path / payload["doc_id"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["doc_id"] == payload["doc_id"]
    assert metadata["filename"] == "sample.txt"
    assert metadata["size_bytes"] == 5


def test_upload_rejects_empty_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "UPLOAD_ROOT", tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )

    assert response.status_code == 400


def test_upload_rejects_extension(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "UPLOAD_ROOT", tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/upload",
        files={"file": ("bad.exe", io.BytesIO(b"hello"), "application/octet-stream")},
    )

    assert response.status_code == 400
