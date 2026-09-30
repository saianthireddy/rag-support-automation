"""Embedding backends behind a single interface.

`OpenAIEmbedder` is used in production; `HashingEmbedder` is a fast,
dependency-free fallback used in tests and offline demos.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

# Function words carry no topic signal. Left in, they dominate a bag-of-words
# vector: "What is the capital of France?" matched support docs on "what",
# "is" and "the" alone, which made off-topic questions look relevant.
STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be
    been being do does did doing i me my we our you your he she it its they them
    their this that these those what which who whom whose when where why how can
    could should would will shall may might must not no so than too very just about
    into over after before again there here all any some such only own same then
    once s t don""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with punctuation and stopwords removed.

    Splitting on whitespace alone kept punctuation attached, so "device?" and
    "device" hashed to different buckets and never matched.
    """
    return [tok for tok in _TOKEN.findall(text.lower()) if tok not in STOPWORDS]


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (no network, no deps).

    4096 buckets rather than 256: with 256, unrelated words collided often
    enough that "How do I bake sourdough bread?" scored 0.32 against the
    security policy. See ``scripts/eval_retrieval.py`` for the measurements.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
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
