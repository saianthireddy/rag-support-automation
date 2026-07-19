"""CLI: evaluate retrieval quality against a small labeled query set.

Runs entirely offline (HashingEmbedder + InMemoryStore, no API keys needed),
so it can be executed in CI or locally in seconds.

Metrics reported, computed at the *source document* level (a hit counts if a
returned chunk came from the expected source file):
  - Precision@1 -- top result is from the expected source
  - Recall@k    -- any of the top-k results are from the expected source
  - MRR         -- mean reciprocal rank of the first correct-source hit

Usage: python scripts/eval_retrieval.py [data/sample_docs] [--top-k 4]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_support.config import get_settings
from rag_support.embeddings.embedder import HashingEmbedder
from rag_support.ingestion.loader import load_documents
from rag_support.ingestion.splitter import split_document
from rag_support.retrieval.retriever import Retriever
from rag_support.vectorstore.memory_store import InMemoryStore


@dataclass
class EvalQuery:
    query: str
    expected_source: str


# Labeled against data/sample_docs/{device_manual.md, support_sop.md}.
# Add a row here whenever a new sample doc is added.
EVAL_SET: list[EvalQuery] = [
    EvalQuery("How do I safely restart the device?", "device_manual.md"),
    EvalQuery("How long does a full reboot take?", "device_manual.md"),
    EvalQuery("Where in the admin console are firmware updates applied?", "device_manual.md"),
    EvalQuery("Which error code should be escalated to Tier 2?", "support_sop.md"),
    EvalQuery("What should I do if a customer reports data loss?", "support_sop.md"),
    EvalQuery("How are incoming support tickets classified?", "support_sop.md"),
]


def build_retriever(root: str, top_k: int) -> Retriever:
    settings = get_settings()
    embedder = HashingEmbedder()
    store = InMemoryStore()

    chunks = []
    for doc in load_documents(root):
        chunks.extend(split_document(doc, settings.chunk_size, settings.chunk_overlap))
    vectors = embedder.embed([c.text for c in chunks])
    store.add(vectors, [{"text": c.text, "source": c.source} for c in chunks])

    return Retriever(embedder, store, top_k=top_k)


def evaluate(retriever: Retriever, eval_set: list[EvalQuery]) -> dict:
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks: list[float] = []

    for item in eval_set:
        results = retriever.retrieve(item.query)
        sources = [r.source for r in results]

        if sources and sources[0] == item.expected_source:
            hits_at_1 += 1
        if item.expected_source in sources:
            hits_at_k += 1

        rank = next(
            (i + 1 for i, s in enumerate(sources) if s == item.expected_source), None
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    n = len(eval_set)
    return {
        "n": n,
        "precision_at_1": hits_at_1 / n,
        "recall_at_k": hits_at_k / n,
        "mrr": sum(reciprocal_ranks) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/sample_docs")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    retriever = build_retriever(args.root, args.top_k)
    metrics = evaluate(retriever, EVAL_SET)

    print(f"Evaluated {metrics['n']} labeled queries against top-{args.top_k} retrieval\n")
    print(f"{'Metric':<15}{'Score':>8}")
    print(f"{'-' * 23}")
    print(f"{'Precision@1':<15}{metrics['precision_at_1']:>8.2f}")
    print(f"{'Recall@' + str(args.top_k):<15}{metrics['recall_at_k']:>8.2f}")
    print(f"{'MRR':<15}{metrics['mrr']:>8.2f}")


if __name__ == "__main__":
    main()
