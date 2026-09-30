"""Guards the numbers the README quotes, so a change to tokenization, the
embedder or MIN_SCORE that quietly degrades retrieval fails CI."""
import importlib.util
import sys
from pathlib import Path

from rag_support.config import get_settings

_SPEC = importlib.util.spec_from_file_location(
    "eval_retrieval", Path(__file__).resolve().parents[1] / "scripts" / "eval_retrieval.py"
)
ev = importlib.util.module_from_spec(_SPEC)
sys.modules["eval_retrieval"] = ev  # dataclasses need the module registered
_SPEC.loader.exec_module(ev)

ROOT = str(Path(__file__).resolve().parents[1] / "data" / "sample_docs")


def test_no_answerable_question_is_refused_and_most_off_topic_are():
    retriever = ev.build_retriever(ROOT, top_k=4, min_score=get_settings().min_score)
    refusal = ev.evaluate_refusal(retriever, ev.EVAL_SET, ev.OFF_TOPIC)
    assert refusal["wrongly_refused"] == []
    assert refusal["off_topic_refused"] >= 0.8


def test_retrieval_quality_floor():
    metrics = ev.evaluate(ev.build_retriever(ROOT, top_k=4), ev.EVAL_SET)
    assert metrics["precision_at_1"] >= 0.85
    assert metrics["recall_at_k"] == 1.0
