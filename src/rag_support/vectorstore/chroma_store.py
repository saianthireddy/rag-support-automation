"""ChromaDB-backed store: an embedded vector database, no server to run.

Uses an in-memory (ephemeral) client by default, or a persistent one when a
directory is given. The collection uses cosine distance, and scores are
returned as cosine similarity (``1 - distance``) so they sit on the same scale
as ``InMemoryStore`` and ``MIN_SCORE`` means the same thing for both.

Chroma's index is approximate (HNSW). On a corpus this size it returns the
same ranking as exact search; ``scripts/eval_retrieval.py --store chroma``
checks that rather than assuming it.
"""

from __future__ import annotations

import uuid

from .base import SearchResult


class ChromaStore:
    def __init__(self, collection: str | None = None, persist_dir: str | None = None):
        import chromadb  # lazy import: only needed when this backend is chosen
        from chromadb.config import Settings

        settings = Settings(anonymized_telemetry=False)
        self._client = (
            chromadb.PersistentClient(path=persist_dir, settings=settings)
            if persist_dir
            else chromadb.EphemeralClient(settings=settings)
        )
        # A unique name per instance keeps ephemeral stores independent: the
        # ephemeral client shares one in-process backend across instances.
        self._collection = self._client.get_or_create_collection(
            name=collection or f"support-kb-{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
        )

    def __len__(self) -> int:
        return self._collection.count()

    def add(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must be the same length")
        if not vectors:
            return
        start = self._collection.count()
        self._collection.add(
            ids=[str(start + i) for i in range(len(vectors))],
            embeddings=vectors,
            documents=[p.get("text", "") for p in payloads],
            metadatas=[{"source": p.get("source", "")} for p in payloads],
        )

    def search(self, vector: list[float], top_k: int = 4) -> list[SearchResult]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(text=doc or "", source=(meta or {}).get("source", ""), score=1.0 - dist)
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
