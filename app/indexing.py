import json
from pathlib import Path

import faiss
import numpy as np

from app.embeddings import EmbeddingsClient

INDEX_ROOT = Path("data/index")
CHUNKS_ROOT = Path("data/chunks")


def load_chunks(doc_id: str) -> list[dict[str, object]]:
    chunks_path = CHUNKS_ROOT / doc_id / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found for {doc_id}.")
    return [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
    ]


def index_doc(doc_id: str) -> None:
    chunks = load_chunks(doc_id)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = EmbeddingsClient().embed_texts(texts)
    if not embeddings:
        raise ValueError("No embeddings generated.")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(embeddings, dtype="float32")
    index.add(vectors)
    index_dir = INDEX_ROOT / doc_id
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    meta_path = index_dir / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "index": chunk["index"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "source": chunk.get("source", {}),
            }
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def load_index(doc_id: str) -> tuple[faiss.Index, list[dict[str, object]]]:
    index_dir = INDEX_ROOT / doc_id
    index = faiss.read_index(str(index_dir / "index.faiss"))
    meta_path = index_dir / "meta.jsonl"
    metadata = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
    return index, metadata
