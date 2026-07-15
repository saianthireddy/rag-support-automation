"""FastAPI application exposing the RAG pipeline.

Run locally:  uvicorn rag_support.api.main:app --reload
"""

from functools import lru_cache

from fastapi import FastAPI

from .. import __version__
from ..config import get_settings
from ..embeddings.embedder import HashingEmbedder
from ..generation.chain import RagChain
from ..ingestion.loader import load_documents
from ..ingestion.splitter import split_document
from ..retrieval.retriever import Retriever
from ..vectorstore.memory_store import InMemoryStore
from .schemas import AskRequest, AskResponse, HealthResponse

app = FastAPI(title="RAG Support Automation", version=__version__)


@lru_cache(maxsize=1)
def get_chain() -> RagChain:
    """Build the pipeline once. Uses offline components by default so the
    service boots without API keys; swap in OpenAIEmbedder / FaissStore /
    PineconeStore via config for production."""
    settings = get_settings()
    embedder = HashingEmbedder()
    store = InMemoryStore()

    chunks = []
    try:
        for doc in load_documents("data/sample_docs"):
            chunks.extend(split_document(doc, settings.chunk_size, settings.chunk_overlap))
    except FileNotFoundError:
        pass
    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        store.add(vectors, [{"text": c.text, "source": c.source} for c in chunks])

    retriever = Retriever(embedder, store, top_k=settings.top_k)
    llm = None if settings.openai_api_key else _offline_llm
    return RagChain(retriever, llm=llm, chat_model=settings.chat_model)


def _offline_llm(system: str, user: str) -> str:
    return (
        "OPENAI_API_KEY is not configured — returning retrieved context only.\n\n" + user
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = get_chain().ask(request.question)
    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        context_used=result.context_used,
    )
