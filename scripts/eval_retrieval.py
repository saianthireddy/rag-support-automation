"""CLI: evaluate retrieval quality against a labeled query set.

Runs entirely offline (HashingEmbedder + InMemoryStore, no API keys needed),
so it can be executed in CI or locally in seconds.

Metrics reported, computed at the *source document* level (a hit counts if a
returned chunk came from the expected source file):
  - Precision@1 -- top result is from the expected source
  - Recall@k    -- any of the top-k results are from the expected source
  - MRR         -- mean reciprocal rank of the first correct-source hit

On making this a real benchmark
------------------------------
An earlier version of this eval scored 1.00 on all three metrics, which looked
good and measured nothing. The corpus was two documents that fit in one chunk
each, so with top_k=4 the retriever returned *the entire index* for every
query -- Recall@4 was arithmetically forced to 1.00 and could never fail. The
six queries also reused the target document's own wording, so ranking was a
two-way choice between documents on unrelated topics.

Two things fix that, and both matter:

  1. Distractors. The corpus now spans five documents that deliberately share
     vocabulary -- "restart", "error code", "admin console" and "data loss"
     each appear in more than one file -- so lexical overlap no longer implies
     the right source.
  2. Paraphrase. Several queries describe a problem in a user's words without
     reusing the document's terms ("the unit is frozen and unresponsive"
     against a manual that says "hold the power button"), which is what real
     support traffic looks like and what a bag-of-words embedder is worst at.

The score is no longer perfect, and that is the point: a benchmark that cannot
fail cannot detect a regression either.

Refusal
-------
Retrieval always returns the nearest chunks, even for "How do I bake sourdough
bread?". ``OFF_TOPIC`` is a set of questions the corpus cannot answer; with the
relevance floor (``MIN_SCORE``) they should come back empty so the chain
escalates. Two numbers matter and pull against each other:
  - off-topic refused  -- higher is better
  - answerable refused -- must stay at 0; each one is a real question dropped
Both sets are small and hand-written, and the floor was chosen on them, so
treat the result as a sanity check rather than a guarantee.

Pipelines and stores
--------------------
``--pipeline native|langchain`` and ``--store memory|chroma`` pick what is
measured; ``--compare`` runs all four combinations side by side. They share the
embedder, chunks, top-k and MIN_SCORE, so a difference between rows is the
framework or the index, not the recipe.

Usage: python scripts/eval_retrieval.py [data/sample_docs] [--top-k 4]
       python scripts/eval_retrieval.py --compare
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
from rag_support.vectorstore.base import SearchResult
from rag_support.vectorstore.memory_store import InMemoryStore


@dataclass
class EvalQuery:
    query: str
    expected_source: str


# Labeled against every file in data/sample_docs/.
# Add a row here whenever a new sample doc is added.
EVAL_SET: list[EvalQuery] = [
    # -- direct: query reuses the source document's vocabulary --------------
    EvalQuery("How do I safely restart the device?", "device_manual.md"),
    EvalQuery("How long does a full reboot take?", "device_manual.md"),
    EvalQuery("Where in the admin console are firmware updates applied?", "device_manual.md"),
    EvalQuery("Which error code should be escalated to Tier 2?", "support_sop.md"),
    EvalQuery("What should I do if a customer reports data loss?", "support_sop.md"),
    EvalQuery("How are incoming support tickets classified?", "support_sop.md"),
    # -- paraphrased: the user's words, not the document's -------------------
    EvalQuery("The unit is frozen and unresponsive, how do I power cycle it?", "device_manual.md"),
    EvalQuery("Roughly how many seconds before the box is usable again?", "device_manual.md"),
    EvalQuery("Who handles a ticket about a broken component?", "support_sop.md"),
    # -- distractor-sensitive: shared terms across several documents ---------
    EvalQuery(
        "A steady amber light on the rear port, what does it mean?", "network_troubleshooting.md"
    ),
    EvalQuery("Customer says DNS is not resolving", "network_troubleshooting.md"),
    EvalQuery("Where do I set a fixed IP address?", "network_troubleshooting.md"),
    EvalQuery("Can I get money back for a charge from last week?", "billing_faq.md"),
    EvalQuery("What happens after a card is declined repeatedly?", "billing_faq.md"),
    EvalQuery("When are statements issued?", "billing_faq.md"),
    EvalQuery(
        "Someone emailed asking me to reset an admin password urgently", "security_policy.md"
    ),
    EvalQuery("We think an attacker got into the account", "security_policy.md"),
    EvalQuery("How long do you keep customer telemetry?", "security_policy.md"),
]


# Questions no document in data/sample_docs/ can answer.
OFF_TOPIC: list[str] = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the World Cup in 2018?",
    "Recommend a good sci-fi novel",
    "What is the weather tomorrow?",
    "How do I learn to play guitar?",
    "Explain quantum entanglement simply",
    "What's a healthy breakfast?",
    "Translate hello into Spanish",
    "How tall is Mount Everest?",
    "Write a poem about autumn",
    "What time zone is Tokyo in?",
]


def _chunks(root: str) -> list:
    settings = get_settings()
    chunks = []
    for doc in load_documents(root):
        chunks.extend(split_document(doc, settings.chunk_size, settings.chunk_overlap))
    return chunks


class _LangChainRetriever:
    """Presents a LangChain pipeline's real retriever through ``.retrieve()``,
    so ``evaluate`` measures exactly what the LangChain chain would retrieve."""

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def retrieve(self, query: str) -> list[SearchResult]:
        return [
            SearchResult(
                text=d.page_content, source=d.metadata.get("source", ""), score=float("nan")
            )
            for d in self._pipeline.retrieve(query)
        ]


def build_retriever(
    root: str, top_k: int, min_score: float = 0.0, pipeline: str = "native", store: str = "memory"
):
    embedder = HashingEmbedder()
    chunks = _chunks(root)

    if pipeline == "langchain":
        from rag_support.generation.langchain_pipeline import (
            LangChainEmbeddings,
            LangChainRagPipeline,
            build_vectorstore,
        )

        vectorstore = build_vectorstore(chunks, LangChainEmbeddings(embedder), store)
        pipeline_ = LangChainRagPipeline(vectorstore, top_k=top_k, min_score=min_score)
        return _LangChainRetriever(pipeline_)

    if store == "chroma":
        from rag_support.vectorstore.chroma_store import ChromaStore

        index = ChromaStore()
    else:
        index = InMemoryStore()
    vectors = embedder.embed([c.text for c in chunks])
    index.add(vectors, [{"text": c.text, "source": c.source} for c in chunks])
    return Retriever(embedder, index, top_k=top_k, min_score=min_score)


def evaluate(retriever: Retriever, eval_set: list[EvalQuery]) -> dict:
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    misses: list[tuple[str, str, str]] = []

    for item in eval_set:
        results = retriever.retrieve(item.query)
        sources = [r.source for r in results]

        if sources and sources[0] == item.expected_source:
            hits_at_1 += 1
        else:
            misses.append((item.query, item.expected_source, sources[0] if sources else "-"))
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
        "misses": misses,
    }


def evaluate_refusal(retriever: Retriever, eval_set: list[EvalQuery], off_topic: list[str]) -> dict:
    wrongly_refused = [q.query for q in eval_set if not retriever.retrieve(q.query)]
    answered_off_topic = [q for q in off_topic if retriever.retrieve(q)]
    return {
        "off_topic_refused": 1 - len(answered_off_topic) / len(off_topic),
        "answerable_refused": len(wrongly_refused) / len(eval_set),
        "wrongly_refused": wrongly_refused,
        "answered_off_topic": answered_off_topic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/sample_docs")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--min-score",
        type=float,
        default=get_settings().min_score,
        help="relevance floor (MIN_SCORE)",
    )
    parser.add_argument("--pipeline", choices=["native", "langchain"], default="native")
    parser.add_argument("--store", choices=["memory", "chroma"], default="memory")
    parser.add_argument(
        "--compare", action="store_true", help="run every pipeline x store combination"
    )
    parser.add_argument(
        "--show-misses", action="store_true", help="list queries whose top hit was wrong"
    )
    args = parser.parse_args()

    if args.compare:
        compare(args.root, args.top_k, args.min_score)
        return

    # Ranking metrics use no floor (a refusal is not a ranking miss); refusal
    # metrics use the floor. Same split as the tests.
    metrics = evaluate(
        build_retriever(args.root, args.top_k, 0.0, args.pipeline, args.store), EVAL_SET
    )
    retriever = build_retriever(args.root, args.top_k, args.min_score, args.pipeline, args.store)
    refusal = evaluate_refusal(retriever, EVAL_SET, OFF_TOPIC)

    print(
        f"Evaluated {metrics['n']} labeled queries against top-{args.top_k} retrieval "
        f"(pipeline={args.pipeline}, store={args.store})\n"
    )
    print(f"{'Metric':<15}{'Score':>8}")
    print(f"{'-' * 23}")
    print(f"{'Precision@1':<15}{metrics['precision_at_1']:>8.2f}")
    print(f"{'Recall@' + str(args.top_k):<15}{metrics['recall_at_k']:>8.2f}")
    print(f"{'MRR':<15}{metrics['mrr']:>8.2f}")
    print(f"\nRefusal at MIN_SCORE={args.min_score:g} ({len(OFF_TOPIC)} off-topic questions)\n")
    print(f"{'Off-topic refused':<24}{refusal['off_topic_refused']:>8.2f}")
    print(f"{'Answerable refused':<24}{refusal['answerable_refused']:>8.2f}")

    if args.show_misses and metrics["misses"]:
        print(f"\nTop-1 misses ({len(metrics['misses'])}):")
        for query, expected, got in metrics["misses"]:
            print(f"  {query}")
            print(f"    expected {expected}, got {got}")
    if args.show_misses and refusal["answered_off_topic"]:
        print(f"\nOff-topic questions still answered ({len(refusal['answered_off_topic'])}):")
        for query in refusal["answered_off_topic"]:
            print(f"  {query}")


def compare(root: str, top_k: int, min_score: float) -> None:
    header = f"{'Pipeline':<11}{'Store':<8}{'P@1':>6}{'R@' + str(top_k):>6}{'MRR':>6}"
    header += f"{'Off-topic refused':>19}{'Answerable refused':>20}"
    print(f"All pipeline x store combinations, top-{top_k}, MIN_SCORE={min_score:g}\n")
    print(header)
    print("-" * len(header))
    for pipeline in ("native", "langchain"):
        for store in ("memory", "chroma"):
            m = evaluate(build_retriever(root, top_k, 0.0, pipeline, store), EVAL_SET)
            r = evaluate_refusal(
                build_retriever(root, top_k, min_score, pipeline, store), EVAL_SET, OFF_TOPIC
            )
            print(
                f"{pipeline:<11}{store:<8}{m['precision_at_1']:>6.2f}{m['recall_at_k']:>6.2f}"
                f"{m['mrr']:>6.2f}{r['off_topic_refused']:>19.2f}{r['answerable_refused']:>20.2f}"
            )


if __name__ == "__main__":
    main()
