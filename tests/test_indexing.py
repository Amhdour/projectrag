from pathlib import Path

import numpy as np

import app.embeddings as embeddings
import app.indexing as indexing
import app.main as main


def test_index_and_search_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_DISABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_DIMENSIONS", "4")
    monkeypatch.setenv("CHUNK_SIZE", "10")
    monkeypatch.setenv("CHUNK_OVERLAP", "2")
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(main, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(indexing, "CHUNKS_ROOT", tmp_path / "chunks")
    monkeypatch.setattr(indexing, "INDEX_ROOT", tmp_path / "index")

    doc_id = "doc-idx"
    processed_dir = main.PROCESSED_ROOT / doc_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "text.txt").write_text("hello world this is a test", encoding="utf-8")

    chunks = main.build_chunks(doc_id)
    indexing.index_doc(doc_id)
    index, metadata = indexing.load_index(doc_id)

    client = embeddings.EmbeddingsClient()
    query_vector = np.array([client.embed_texts([chunks[0]["text"]])[0]], dtype="float32")
    distances, indices = index.search(query_vector, 1)

    assert distances.shape == (1, 1)
    assert indices[0][0] >= 0
    assert metadata[indices[0][0]]["chunk_id"] == chunks[0]["chunk_id"]
