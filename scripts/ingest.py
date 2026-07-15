"""CLI: ingest a folder of docs and report chunk statistics.

Usage: python scripts/ingest.py data/sample_docs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_support.config import get_settings
from rag_support.embeddings.embedder import HashingEmbedder
from rag_support.ingestion.loader import load_documents
from rag_support.ingestion.splitter import split_document
from rag_support.vectorstore.memory_store import InMemoryStore


def main(root: str) -> None:
    settings = get_settings()
    embedder = HashingEmbedder()
    store = InMemoryStore()

    total_docs = 0
    chunks = []
    for doc in load_documents(root):
        total_docs += 1
        chunks.extend(split_document(doc, settings.chunk_size, settings.chunk_overlap))

    vectors = embedder.embed([c.text for c in chunks])
    store.add(vectors, [{"text": c.text, "source": c.source} for c in chunks])

    print(f"Ingested {total_docs} documents -> {len(chunks)} chunks -> {len(store)} vectors")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs")
