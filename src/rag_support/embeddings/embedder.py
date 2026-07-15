"""Embedding backends behind a single interface.

`OpenAIEmbedder` is used in production; `HashingEmbedder` is a fast,
dependency-free fallback used in tests and offline demos.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (no network, no deps)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAIEmbedder:
    """Thin wrapper over the OpenAI embeddings API (lazy import)."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        from openai import OpenAI  # imported lazily so tests never need it

        self._client = OpenAI()
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
