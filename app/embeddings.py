import hashlib
import os
import time
from typing import Iterable

import httpx

DEFAULT_PROVIDER = "openai"
DEFAULT_DIMENSIONS = 8


class EmbeddingsClient:
    def __init__(self) -> None:
        self.provider = os.getenv("EMBEDDINGS_PROVIDER", DEFAULT_PROVIDER)
        self.model = os.getenv("EMBEDDINGS_MODEL", "")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.disabled = os.getenv("EMBEDDINGS_DISABLED", "false").lower() == "true"
        self.dimensions = int(os.getenv("EMBEDDINGS_DIMENSIONS", str(DEFAULT_DIMENSIONS)))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.disabled:
            return [self._fake_vector(text) for text in texts]
        if self.provider != "openai":
            raise ValueError(f"Unsupported embeddings provider: {self.provider}")
        if not self.model:
            raise ValueError("EMBEDDINGS_MODEL is required when embeddings are enabled.")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when embeddings are enabled.")
        return self._embed_with_openai(texts)

    def _fake_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for index in range(self.dimensions):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) * 2 - 1)
        return values

    def _embed_with_openai(self, texts: list[str]) -> list[list[float]]:
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }
        attempts = 3
        backoff = 0.5
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=30) as client:
                    response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data.get("data", [])]
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Failed to fetch embeddings from provider.") from last_error


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    client = EmbeddingsClient()
    return client.embed_texts(list(texts))
