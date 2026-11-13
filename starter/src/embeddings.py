"""Real embedding provider adapters (Local SentenceTransformers & Google Gemini API)."""
from __future__ import annotations

import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from google import genai
from google.genai import types


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    msg = str(exc)
    m = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m2 = re.search(r"'retryDelay':\s*'([0-9]+)s'", msg)
    if m2:
        return float(m2.group(1))
    return None


@dataclass
class EmbeddingUsage:
    provider: str = ""
    model_name: str = ""
    api_calls: int = 0
    texts_embedded: int = 0
    total_chars: int = 0
    elapsed_seconds: float = 0.0
    rate_limit_waits: int = 0
    errors: list[str] = field(default_factory=list)


class LocalSentenceTransformerEmbedder:
    """Local real dense embedder using SentenceTransformers (no API key or rate limit)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 32,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix if "bge" in model_name.lower() else ""
        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        if not dim:
            raise RuntimeError(f"Could not determine embedding dimension for {model_name}")
        self.dimensions = int(dim)
        self.usage = EmbeddingUsage(provider="local_sentence_transformers", model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        arr = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - t0
        vectors = [[float(x) for x in row] for row in arr]
        self.usage.api_calls += math.ceil(len(texts) / self.batch_size)
        self.usage.texts_embedded += len(texts)
        self.usage.total_chars += sum(len(t) for t in texts)
        self.usage.elapsed_seconds += elapsed
        return vectors

    def embed_query(self, question: str) -> list[float]:
        t0 = time.perf_counter()
        q_input = f"{self.query_prefix}{question}" if self.query_prefix else question
        arr = self._model.encode(
            [q_input],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - t0
        self.usage.api_calls += 1
        self.usage.texts_embedded += 1
        self.usage.total_chars += len(question)
        self.usage.elapsed_seconds += elapsed
        return [float(x) for x in arr[0]]


class GeminiEmbedder:
    """Gemini embedding adapter with Free-Tier RPM pacing and explicit error propagation."""

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        dimensions: int = 768,
        batch_size: int = 45,
        max_retries: int = 4,
        rpm_item_budget: int = 90,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for GeminiEmbedder. "
                "Do not fall back to mock vectors."
            )
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.rpm_item_budget = rpm_item_budget
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30000),
        )
        self.usage = EmbeddingUsage(provider="google_gemini_api", model_name=model_name)
        self._window: deque[tuple[float, int]] = deque()

    def _pace(self, n_items: int) -> None:
        if self.rpm_item_budget <= 0:
            return
        while True:
            now = time.monotonic()
            while self._window and now - self._window[0][0] >= 62.0:
                self._window.popleft()
            used = sum(cnt for _, cnt in self._window)
            if used + n_items <= self.rpm_item_budget or not self._window:
                return
            wait_s = 62.5 - (now - self._window[0][0])
            if wait_s > 0:
                self.usage.rate_limit_waits += 1
                time.sleep(wait_s)

    def _call_with_retry(self, contents: list[str], task_type: str) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._pace(len(contents))
            t0 = time.perf_counter()
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimensions,
                    ),
                )
                elapsed = time.perf_counter() - t0
                self._window.append((time.monotonic(), len(contents)))
                vectors = [_l2_normalize(list(emb.values)) for emb in response.embeddings]
                if len(vectors) != len(contents):
                    raise RuntimeError(f"Expected {len(contents)} embeddings, got {len(vectors)}")
                for v in vectors:
                    if len(v) != self.dimensions:
                        raise RuntimeError(f"Expected {self.dimensions} dims, got {len(v)}")
                self.usage.api_calls += 1
                self.usage.texts_embedded += len(contents)
                self.usage.total_chars += sum(len(c) for c in contents)
                self.usage.elapsed_seconds += elapsed
                return vectors
            except Exception as exc:
                last_exc = exc
                self.usage.errors.append(f"attempt {attempt + 1}: {str(exc)[:180]}")
                retry_hint = _extract_retry_delay_seconds(exc)
                # Never sleep for multi-hour daily quota exhaustion
                if retry_hint is not None and retry_hint > 120.0:
                    raise RuntimeError(
                        f"Daily/long quota exhausted (retryDelay={retry_hint}s): {exc}"
                    ) from exc
                if attempt < self.max_retries - 1:
                    sleep_s = (retry_hint + 2.0) if retry_hint is not None else min(65.0, 4.0 * (2 ** attempt))
                    self.usage.rate_limit_waits += 1
                    time.sleep(sleep_s)
        raise RuntimeError(f"Gemini embedding request failed after {self.max_retries} attempts: {last_exc}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_vectors.extend(self._call_with_retry(batch, task_type="RETRIEVAL_DOCUMENT"))
        return all_vectors

    def embed_query(self, question: str) -> list[float]:
        vectors = self._call_with_retry([question], task_type="RETRIEVAL_QUERY")
        return vectors[0]


def build_embedder(provider: str = "local", model_name: str | None = None) -> Embedder:
    """Create configured real embedder; never silently falls back to mock."""
    if provider == "gemini":
        return GeminiEmbedder(model_name=model_name or "gemini-embedding-001", dimensions=768)
    if provider == "local":
        return LocalSentenceTransformerEmbedder(model_name=model_name or "BAAI/bge-base-en-v1.5")
    raise ValueError(f"Unsupported embedding provider: {provider!r}")


MyEmbedder = LocalSentenceTransformerEmbedder
