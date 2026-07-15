"""Pure-python cosine-similarity store — zero dependencies, great for tests."""

import math

from .base import SearchResult


class InMemoryStore:
    def __init__(self):
        self._vectors: list[list[float]] = []
        self._payloads: list[dict] = []

    def add(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must be the same length")
        self._vectors.extend(vectors)
        self._payloads.extend(payloads)

    def __len__(self) -> int:
        return len(self._vectors)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def search(self, vector: list[float], top_k: int = 4) -> list[SearchResult]:
        scored = [
            (self._cosine(vector, v), payload)
            for v, payload in zip(self._vectors, self._payloads)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(text=p.get("text", ""), source=p.get("source", ""), score=s)
            for s, p in scored[:top_k]
        ]
