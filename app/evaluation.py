import json
from pathlib import Path

import numpy as np

from app.embeddings import EmbeddingsClient
from app.indexing import load_chunks, load_index


def run_eval(path: str) -> dict[str, object]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Evaluation file must be a JSON array.")

    per_item = []
    total_recall = 0
    for item in items:
        doc_id = item["doc_id"]
        question = item["question"]
        gold_chunk_ids = set(item.get("gold_chunk_ids", []))
        k = int(item.get("k", 5))

        query_vector = EmbeddingsClient().embed_texts([question])[0]
        index, metadata = load_index(doc_id)
        chunks = load_chunks(doc_id)
        chunk_texts = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}

        distances, indices = index.search(np.array([query_vector], dtype="float32"), k)
        retrieved_ids = []
        for position in indices[0]:
            if position < 0 or position >= len(metadata):
                continue
            retrieved_ids.append(metadata[position]["chunk_id"])
        recall = 1 if gold_chunk_ids.intersection(retrieved_ids) else 0
        total_recall += recall
        per_item.append(
            {
                "doc_id": doc_id,
                "question": question,
                "k": k,
                "recall": recall,
                "retrieved_chunk_ids": retrieved_ids,
                "gold_chunk_ids": list(gold_chunk_ids),
            }
        )

    total = len(items)
    avg_recall = total_recall / total if total else 0
    return {"total": total, "avg_recall": avg_recall, "per_item": per_item}
