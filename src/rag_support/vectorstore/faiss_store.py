"""FAISS-backed store for local / self-hosted deployments."""

from .base import SearchResult


class FaissStore:
    def __init__(self, dim: int):
        import faiss  # lazy import
        import numpy as np

        self._faiss = faiss
        self._np = np
        self._index = faiss.IndexFlatIP(dim)
        self._payloads: list[dict] = []

    def add(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        array = self._np.asarray(vectors, dtype="float32")
        self._faiss.normalize_L2(array)
        self._index.add(array)
        self._payloads.extend(payloads)

    def search(self, vector: list[float], top_k: int = 4) -> list[SearchResult]:
        query = self._np.asarray([vector], dtype="float32")
        self._faiss.normalize_L2(query)
        scores, ids = self._index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            payload = self._payloads[idx]
            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    source=payload.get("source", ""),
                    score=float(score),
                )
            )
        return results
