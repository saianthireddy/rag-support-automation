"""Retriever: embeds the query and returns the most relevant chunks."""

from ..embeddings.embedder import Embedder
from ..vectorstore.base import SearchResult, VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 4):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def retrieve(self, query: str) -> list[SearchResult]:
        if not query.strip():
            return []
        vector = self._embedder.embed([query])[0]
        return self._store.search(vector, top_k=self._top_k)
