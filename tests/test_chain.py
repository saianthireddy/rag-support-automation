from rag_support.generation.chain import RagChain
from tests.test_retrieval import build_retriever


def fake_llm(system: str, user: str) -> str:
    assert "Context:" in user
    return "Hold the power button for ten seconds. [router.md]"


def test_chain_answers_with_sources():
    chain = RagChain(build_retriever(), llm=fake_llm)
    result = chain.ask("How do I reset the router?")
    assert "[router.md]" in result.answer
    assert result.context_used == 2
    assert "router.md" in result.sources


def test_chain_handles_no_results():
    chain = RagChain(build_retriever(), llm=fake_llm)
    result = chain.ask("")
    assert result.sources == []
    assert "escalate" in result.answer.lower()
