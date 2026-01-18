import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

from app.embeddings import EmbeddingsClient
from app.evaluation import run_eval
from app.indexing import index_doc, load_chunks, load_index

app = FastAPI()
logger = logging.getLogger("projectrag.uploads")
query_logger = logging.getLogger("projectrag.query")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
UPLOAD_ROOT = Path("data/uploads")
PROCESSED_ROOT = Path("data/processed")
CHUNKS_ROOT = Path("data/chunks")
LOGS_ROOT = Path("data/logs")


def max_upload_size_bytes() -> int:
    max_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    return max_mb * 1024 * 1024


def chunk_size() -> int:
    return int(os.getenv("CHUNK_SIZE", "1200"))


def chunk_overlap() -> int:
    return int(os.getenv("CHUNK_OVERLAP", "200"))


def min_retrieval_score() -> float:
    return float(os.getenv("MIN_RETRIEVAL_SCORE", "-0.1"))


def log_event(event: dict[str, object]) -> None:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = LOGS_ROOT / f"{date_str}.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def validate_upload(upload: UploadFile) -> None:
    filename = upload.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file extension.")


def normalize_text(raw_text: str) -> str:
    normalized = raw_text.replace("\r\n", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    collapsed_lines = []
    empty_streak = 0
    for line in lines:
        if line == "":
            empty_streak += 1
            if empty_streak <= 2:
                collapsed_lines.append(line)
        else:
            empty_streak = 0
            collapsed_lines.append(line)
    return "\n".join(collapsed_lines)


def extract_text_from_file(file_path: Path) -> str:
    raw_text = file_path.read_text(encoding="utf-8")
    return normalize_text(raw_text)


def save_processed_text(doc_id: str, text: str) -> Path:
    processed_dir = PROCESSED_ROOT / doc_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    text_path = processed_dir / "text.txt"
    text_path.write_text(text, encoding="utf-8")
    return text_path


def extract_pdf_pages(file_path: Path) -> list[dict[str, str | int]]:
    reader = PdfReader(str(file_path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append({"page": index, "text": normalize_text(page_text)})
    return pages


def save_processed_pages(doc_id: str, pages: list[dict[str, str | int]]) -> Path:
    processed_dir = PROCESSED_ROOT / doc_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    pages_path = processed_dir / "pages.json"
    pages_path.write_text(json.dumps(pages, indent=2), encoding="utf-8")
    return pages_path


def format_pdf_text(pages: list[dict[str, str | int]]) -> str:
    parts = []
    for page in pages:
        parts.append(f"--- Page {page['page']} ---")
        parts.append(str(page["text"]))
    return "\n\n".join(parts)


def build_pdf_text_and_ranges(
    pages: list[dict[str, str | int]],
) -> tuple[str, list[dict[str, int]]]:
    pieces = []
    ranges = []
    cursor = 0
    for index, page in enumerate(pages):
        if index > 0:
            pieces.append("\n\n")
            cursor += 2
        header = f"--- Page {page['page']} ---\n\n"
        pieces.append(header)
        cursor += len(header)
        text = str(page["text"])
        start = cursor
        pieces.append(text)
        cursor += len(text)
        ranges.append(
            {
                "page": int(page["page"]),
                "start": start,
                "end": cursor,
            }
        )
    return "".join(pieces), ranges


def build_chunks(doc_id: str) -> list[dict[str, object]]:
    processed_dir = PROCESSED_ROOT / doc_id
    text_path = processed_dir / "text.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"Processed text not found for {doc_id}.")
    text = text_path.read_text(encoding="utf-8")
    pages_path = processed_dir / "pages.json"
    page_ranges: list[dict[str, int]] = []
    if pages_path.exists():
        pages = json.loads(pages_path.read_text(encoding="utf-8"))
        text, page_ranges = build_pdf_text_and_ranges(pages)
        if text != text_path.read_text(encoding="utf-8"):
            text_path.write_text(text, encoding="utf-8")
    size = chunk_size()
    overlap = chunk_overlap()
    if size <= 0:
        raise ValueError("CHUNK_SIZE must be positive.")
    if overlap < 0 or overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be between 0 and CHUNK_SIZE-1.")
    chunks: list[dict[str, object]] = []
    start = 0
    index = 0
    step = size - overlap
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end]
        chunk = {
            "chunk_id": f"{doc_id}:{index}",
            "doc_id": doc_id,
            "index": index,
            "start_char": start,
            "end_char": end,
            "text": chunk_text,
            "source": {},
        }
        if page_ranges:
            page_start = None
            page_end = None
            for page in page_ranges:
                if page["end"] <= start or page["start"] >= end:
                    continue
                if page_start is None:
                    page_start = page["page"]
                page_end = page["page"]
            if page_start is not None:
                chunk["source"] = {
                    "page_start": page_start,
                    "page_end": page_end,
                }
        chunks.append(chunk)
        index += 1
        start += step
    chunks_dir = CHUNKS_ROOT / doc_id
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = chunks_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return chunks


def write_upload(upload: UploadFile, destination: Path) -> int:
    max_bytes = max_upload_size_bytes()
    bytes_written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                buffer.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds size limit.")
            buffer.write(chunk)
    if bytes_written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is empty.")
    return bytes_written


class QueryRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 5
    answer: bool = True


class EvalRequest(BaseModel):
    path: str


@app.get("/health")
def health_check() -> dict[str, bool]:
    return {"ok": True}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)) -> dict[str, str]:
    trace_id = str(uuid4())
    start_time = time.perf_counter()
    validate_upload(file)
    doc_id = str(uuid4())
    filename = file.filename or "upload"
    destination = UPLOAD_ROOT / doc_id / filename
    size_bytes = write_upload(file, destination)
    metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "content_type": file.content_type or "",
        "size_bytes": size_bytes,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = destination.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    status = "stored"
    suffix = destination.suffix.lower()
    if suffix in {".txt", ".md"}:
        extracted_text = extract_text_from_file(destination)
        save_processed_text(doc_id, extracted_text)
        build_chunks(doc_id)
        index_doc(doc_id)
        status = "processed_text"
    if suffix == ".pdf":
        pages = extract_pdf_pages(destination)
        save_processed_pages(doc_id, pages)
        pdf_text = format_pdf_text(pages)
        save_processed_text(doc_id, pdf_text)
        build_chunks(doc_id)
        index_doc(doc_id)
        status = "processed_pdf"
        total_chars = sum(len(str(page["text"])) for page in pages)
        logger.info(
            "Extracted %s pages (%s chars) for %s", len(pages), total_chars, doc_id
        )
    duration_ms = (time.perf_counter() - start_time) * 1000
    log_event(
        {
            "trace_id": trace_id,
            "doc_id": doc_id,
            "event_type": "upload",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "content_type": file.content_type or "",
            "size_bytes": size_bytes,
            "status": status,
            "duration_ms": duration_ms,
        }
    )
    logger.info("Stored upload %s (%s bytes) at %s", doc_id, size_bytes, destination)
    return {"doc_id": doc_id, "filename": filename, "status": status, "trace_id": trace_id}


@app.post("/query")
def query_index(request: QueryRequest) -> dict[str, object]:
    trace_id = str(uuid4())
    question = request.question
    query_logger.info(
        "Query received %s (length=%s)", trace_id, len(question)
    )
    embed_start = time.perf_counter()
    query_vector = EmbeddingsClient().embed_texts([question])[0]
    embed_time = (time.perf_counter() - embed_start) * 1000

    index, metadata = load_index(request.doc_id)
    chunks = load_chunks(request.doc_id)
    chunk_texts = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}

    search_start = time.perf_counter()
    distances, indices = index.search(
        np.array([query_vector], dtype="float32"),
        request.top_k,
    )
    search_time = (time.perf_counter() - search_start) * 1000

    results = []
    returned_ids = []
    scores = []
    for rank, position in enumerate(indices[0], start=1):
        if position < 0 or position >= len(metadata):
            continue
        meta = metadata[position]
        chunk_id = meta["chunk_id"]
        returned_ids.append(chunk_id)
        text = chunk_texts.get(chunk_id, "")
        similarity = -float(distances[0][rank - 1])
        scores.append(similarity)
        results.append(
            {
                "rank": rank,
                "score": similarity,
                "chunk_id": chunk_id,
                "source": meta.get("source", {}),
                "text_preview": text[:200],
            }
        )

    answer_text = ""
    citations: list[dict[str, object]] = []
    if request.answer:
        if not results or max(scores, default=float("-inf")) < min_retrieval_score():
            answer_text = "Insufficient evidence in provided documents."
        else:
            answer_lines = []
            for result in results:
                chunk_id = result["chunk_id"]
                text = chunk_texts.get(chunk_id, "")
                answer_lines.append(f"Chunk {chunk_id}:\n{text}")
                meta = next(
                    (item for item in metadata if item["chunk_id"] == chunk_id), None
                )
                if meta:
                    citations.append(
                        {
                            "chunk_id": chunk_id,
                            "doc_id": meta["doc_id"],
                            "source": meta.get("source", {}),
                            "start_char": meta["start_char"],
                            "end_char": meta["end_char"],
                        }
                    )
            answer_text = "\n\n".join(answer_lines)

    query_logger.info(
        "Query %s completed (embed_ms=%.2f, search_ms=%.2f, chunks=%s)",
        trace_id,
        embed_time,
        search_time,
        returned_ids,
    )
    log_event(
        {
            "trace_id": trace_id,
            "doc_id": request.doc_id,
            "event_type": "query",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_length": len(question),
            "top_k": request.top_k,
            "embed_ms": embed_time,
            "search_ms": search_time,
            "returned_chunk_ids": returned_ids,
            "answer_included": request.answer,
        }
    )
    return {
        "answer": answer_text,
        "citations": citations,
        "doc_id": request.doc_id,
        "question": question,
        "top_k": request.top_k,
        "results": results,
        "trace_id": trace_id,
    }


@app.post("/eval/run")
def run_evaluation(request: EvalRequest) -> dict[str, object]:
    return run_eval(request.path)


@app.get("/logs")
def get_logs(
    n: int = 100,
    trace_id: str | None = None,
    doc_id: str | None = None,
) -> dict[str, object]:
    if n <= 0:
        raise HTTPException(status_code=400, detail="n must be positive.")
    if not LOGS_ROOT.exists():
        return {"events": []}
    log_files = sorted(LOGS_ROOT.glob("*.jsonl"))
    events: list[dict[str, object]] = []
    for log_file in log_files:
        lines = log_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if trace_id and event.get("trace_id") != trace_id:
                continue
            if doc_id and event.get("doc_id") != doc_id:
                continue
            events.append(event)
    return {"events": events[-n:]}
