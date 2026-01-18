#!/usr/bin/env bash
set -euo pipefail

export EMBEDDINGS_DISABLED=true
export EMBEDDINGS_DIMENSIONS=8
export MIN_RETRIEVAL_SCORE=-1.0

uvicorn app.main:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
trap 'kill ${SERVER_PID}' EXIT

sleep 1

UPLOAD_RESPONSE=$(curl -s -F "file=@data/samples/sample.txt" http://127.0.0.1:8000/upload)
export UPLOAD_RESPONSE
DOC_ID=$(python - <<'PY'
import json
import os
payload = json.loads(os.environ["UPLOAD_RESPONSE"])
print(payload["doc_id"])
PY
)
export DOC_ID

QUERY_RESPONSE=$(curl -s -H "Content-Type: application/json" \
  -d "{\"doc_id\": \"${DOC_ID}\", \"question\": \"sample document\", \"top_k\": 1}" \
  http://127.0.0.1:8000/query)
export QUERY_RESPONSE

python - <<'PY'
import json
import os
payload = json.loads(os.environ["QUERY_RESPONSE"])
print("Answer:\n" + payload["answer"])
print("\nCitations:")
for cite in payload["citations"]:
    print(f"- {cite['chunk_id']}")
PY

python - <<'PY'
import json
from pathlib import Path
import os

doc_id = os.environ["DOC_ID"]
chunks_path = Path("data/chunks") / doc_id / "chunks.jsonl"
first_chunk = json.loads(chunks_path.read_text(encoding="utf-8").splitlines()[0])["chunk_id"]

eval_path = Path("data/eval") / "sample.json"
eval_path.write_text(json.dumps([
    {
        "doc_id": doc_id,
        "question": "sample document",
        "gold_chunk_ids": [first_chunk],
        "k": 5,
    }
], indent=2), encoding="utf-8")
PY

python -m eval.run --path data/eval/sample.json
