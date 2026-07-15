"""Vector store interface and shared search result type."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SearchResult:
    text: str
    source: str
    score: float


class VectorStore(Protocol):
    def add(self, vectors: list[list[float]], payloads: list[dict]) -> None: ...

    def search(self, vector: list[float], top_k: int = 4) -> list[SearchResult]: ...
