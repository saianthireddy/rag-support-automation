"""The LangChain pipeline and the Chroma store must behave exactly like the
native pipeline and in-memory store they sit beside: same retrieval, same
refusals, same escalation. These tests pin that, so the README's comparison
table cannot quietly drift."""

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

from rag_support.api.main import build_pipeline
from rag_support.config import get_settings
from rag_support.embeddings.embedder import HashingEmbedder
from rag_support.generation.chain import ESCALATION
from rag_support.generation.langchain_pipeline import (
    LangChainEmbeddings,
    LangChainRagPipeline,
    build_vectorstore,
)
from rag_support.ingestion.splitter import Chunk
from rag_support.vectorstore.chroma_store import ChromaStore
from rag_support.vectorstore.memory_store import InMemoryStore

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "eval_retrieval", ROOT / "scripts" / "eval_retrieval.py"
)
ev = importlib.util.module_from_spec(_SPEC)
sys.modules["eval_retrieval"] = ev
_SPEC.loader.exec_module(ev)

DOCS = str(ROOT / "data" / "sample_docs")
CHUNKS = [
    Chunk("Restart the device by holding the power button for ten seconds.", "manual.md", 0),
    Chunk("Invoices are issued on the first business day of each month.", "billing.md", 0),
]


def _floor() -> float:
    return get_settings().min_score


# -- Chroma store ---------------------------------------------------------------


def test_chroma_scores_match_the_in_memory_store():
    embedder = HashingEmbedder()
    vectors = embedder.embed([c.text for c in CHUNKS])
    payloads = [{"text": c.text, "source": c.source} for c in CHUNKS]
    memory, chroma = InMemoryStore(), ChromaStore()
    memory.add(vectors, payloads)
    chroma.add(vectors, payloads)

    query = embedder.embed(["how do I restart the device"])[0]
    expected = [(r.source, r.score) for r in memory.search(query, top_k=2)]
    got = [(r.source, r.score) for r in chroma.search(query, top_k=2)]
    assert [s for s, _ in got] == [s for s, _ in expected]
    for (_, a), (_, b) in zip(got, expected):
        assert a == pytest.approx(b, abs=1e-5)


def test_empty_chroma_store_returns_nothing():
    assert ChromaStore().search([1.0, 0.0], top_k=4) == []


# -- LangChain pipeline ---------------------------------------------------------


@pytest.mark.parametrize("backend", ["memory", "chroma"])
def test_langchain_escalates_off_topic_without_calling_the_llm(backend):
    def llm_must_not_run(_prompt):
        raise AssertionError("the LLM must not be called when nothing relevant was retrieved")

    store = build_vectorstore(CHUNKS, LangChainEmbeddings(HashingEmbedder()), backend)
    pipeline = LangChainRagPipeline(store, min_score=_floor(), llm=llm_must_not_run)
    for question in ["What is the capital of France?", "what is it?"]:  # 2nd = all stopwords
        result = pipeline.ask(question)
        assert result.answer == ESCALATION
        assert result.sources == [] and result.context_used == 0


@pytest.mark.parametrize("backend", ["memory", "chroma"])
def test_langchain_sends_grounded_context_to_the_llm(backend):
    seen = {}

    def fake_llm(prompt_value):
        seen["messages"] = prompt_value.to_messages()
        return "Hold the power button for ten seconds. [manual.md]"

    store = build_vectorstore(CHUNKS, LangChainEmbeddings(HashingEmbedder()), backend)
    result = LangChainRagPipeline(store, top_k=1, min_score=_floor(), llm=fake_llm).ask(
        "How do I restart the device?"
    )
    assert result.answer.endswith("[manual.md]")
    assert result.sources == ["manual.md"] and result.context_used == 1
    system, human = seen["messages"]
    assert "Answer ONLY from the provided context" in system.content
    assert "[manual.md]" in human.content and "How do I restart the device?" in human.content


# -- the comparison the README quotes --------------------------------------------


def test_every_pipeline_and_store_scores_the_same():
    """Same embedder, chunks, k and floor: the framework and index must not
    change a single number. If this fails, the README table is wrong."""
    results = {}
    for pipeline in ("native", "langchain"):
        for store in ("memory", "chroma"):
            ranking = ev.evaluate(ev.build_retriever(DOCS, 4, 0.0, pipeline, store), ev.EVAL_SET)
            refusal = ev.evaluate_refusal(
                ev.build_retriever(DOCS, 4, _floor(), pipeline, store), ev.EVAL_SET, ev.OFF_TOPIC
            )
            results[(pipeline, store)] = (
                round(ranking["precision_at_1"], 4),
                round(ranking["recall_at_k"], 4),
                round(ranking["mrr"], 4),
                round(refusal["off_topic_refused"], 4),
                round(refusal["answerable_refused"], 4),
            )
    assert len(set(results.values())) == 1, results


# -- API wiring -------------------------------------------------------------------


@pytest.mark.parametrize(
    "pipeline,backend", [("langchain", "memory"), ("langchain", "chroma"), ("native", "chroma")]
)
def test_api_pipeline_switches(pipeline, backend):
    settings = dataclasses.replace(
        get_settings(), pipeline=pipeline, vector_backend=backend, openai_api_key=""
    )
    chain = build_pipeline(settings)
    off_topic = chain.ask("What is the capital of France?")
    assert off_topic.answer == ESCALATION
    on_topic = chain.ask("How do I restart the device?")
    assert "device_manual.md" in on_topic.sources


def test_unwired_backend_is_refused_clearly():
    settings = dataclasses.replace(get_settings(), vector_backend="pinecone")
    with pytest.raises(ValueError, match="not wired"):
        build_pipeline(settings)
