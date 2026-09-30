"""Retriever: embeds the query and returns the most relevant chunks."""

from ..embeddings.embedder import Embedder
from ..vectorstore.base import SearchResult, VectorStore


class Retriever:
    def __init__(
        self, embedder: Embedder, store: VectorStore, top_k: int = 4, min_score: float = 0.0
    ):
        """``min_score`` drops chunks whose similarity is below it.

        Without a floor, top-k always returns *something*, so an off-topic
        question got the nearest unrelated chunks as "context". With one, it
        gets nothing and the chain escalates instead of answering.
        """
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(self, query: str) -> list[SearchResult]:
        if not query.strip():
            return []
        vector = self._embedder.embed([query])[0]
        results = self._store.search(vector, top_k=self._top_k)
        return [r for r in results if r.score >= self._min_score]
