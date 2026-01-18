import json
from pathlib import Path

import app.main as main


def read_chunks(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_chunking_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(main, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setenv("CHUNK_SIZE", "10")
    monkeypatch.setenv("CHUNK_OVERLAP", "3")

    doc_id = "doc-1"
    processed_dir = main.PROCESSED_ROOT / doc_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "text.txt").write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")

    first = main.build_chunks(doc_id)
    second = main.build_chunks(doc_id)

    assert first == second

    chunks_path = main.CHUNKS_ROOT / doc_id / "chunks.jsonl"
    assert read_chunks(chunks_path) == first


def test_chunk_overlap_correctness(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(main, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setenv("CHUNK_SIZE", "8")
    monkeypatch.setenv("CHUNK_OVERLAP", "3")

    doc_id = "doc-2"
    processed_dir = main.PROCESSED_ROOT / doc_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "text.txt").write_text("abcdefghijklmno", encoding="utf-8")

    chunks = main.build_chunks(doc_id)

    assert len(chunks) >= 2
    step = 8 - 3
    for idx in range(len(chunks) - 1):
        current = chunks[idx]
        next_chunk = chunks[idx + 1]
        assert next_chunk["start_char"] == current["start_char"] + step
        overlap_text = current["text"][-3:]
        assert next_chunk["text"].startswith(overlap_text)
