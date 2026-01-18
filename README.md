# projectrag

Minimal FastAPI service scaffold (no RAG logic yet).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

## Run

```bash
make run
```

The service will be available at http://127.0.0.1:8000/health.

## Test

```bash
make test
```

## Lint

```bash
make lint
```

## Evaluation

Run retrieval evaluation against a JSON file containing items with `doc_id`, `question`,
`gold_chunk_ids`, and optional `k` values:

```bash
python -m eval.run --path data/eval/sample.json
```

## Step-1 End-to-End Demo

Run a script that starts the server, uploads a sample doc, queries it, prints the answer
with citations, and runs a recall@k eval:

```bash
./scripts/step1_demo.sh
```
