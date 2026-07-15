"""Pinecone-backed store for managed cloud deployments."""

import uuid

from .base import SearchResult


class PineconeStore:
    def __init__(self, api_key: str, index_name: str):
        from pinecone import Pinecone  # lazy import

        self._index = Pinecone(api_key=api_key).Index(index_name)

    def add(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        items = [
            {"id": str(uuid.uuid4()), "values": vec, "metadata": payload}
            for vec, payload in zip(vectors, payloads)
        ]
        self._index.upsert(vectors=items)

    def search(self, vector: list[float], top_k: int = 4) -> list[SearchResult]:
        response = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            SearchResult(
                text=match["metadata"].get("text", ""),
                source=match["metadata"].get("source", ""),
                score=float(match["score"]),
            )
            for match in response["matches"]
        ]
