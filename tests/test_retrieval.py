from rag_support.embeddings.embedder import HashingEmbedder
from rag_support.retrieval.retriever import Retriever
from rag_support.vectorstore.memory_store import InMemoryStore

DOCS = [
    ("Reset the router by holding the power button for ten seconds.", "router.md"),
    ("The billing portal accepts credit cards and ACH transfers.", "billing.md"),
    ("Update firmware from the admin console under Settings > System.", "firmware.md"),
]


def build_retriever(top_k: int = 2) -> Retriever:
    embedder = HashingEmbedder(dim=128)
    store = InMemoryStore()
    vectors = embedder.embed([text for text, _ in DOCS])
    store.add(vectors, [{"text": t, "source": s} for t, s in DOCS])
    return Retriever(embedder, store, top_k=top_k)


def test_retrieves_most_relevant_chunk_first():
    results = build_retriever().retrieve("how do I reset the router power button")
    assert results[0].source == "router.md"


def test_top_k_respected():
    assert len(build_retriever(top_k=2).retrieve("firmware update")) == 2


def test_empty_query_returns_nothing():
    assert build_retriever().retrieve("   ") == []


def test_scores_are_sorted_descending():
    results = build_retriever(top_k=3).retrieve("billing credit card payment")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_tokenizer_drops_punctuation_and_stopwords():
    from rag_support.embeddings.embedder import tokenize

    assert tokenize("How do I restart the device?") == ["restart", "device"]


def test_min_score_filters_weak_matches():
    embedder = HashingEmbedder(dim=128)
    store = InMemoryStore()
    store.add(embedder.embed([t for t, _ in DOCS]), [{"text": t, "source": s} for t, s in DOCS])
    strict = Retriever(embedder, store, top_k=3, min_score=0.2)
    assert strict.retrieve("capital of France") == []
    assert strict.retrieve("reset the router power button")[0].source == "router.md"
