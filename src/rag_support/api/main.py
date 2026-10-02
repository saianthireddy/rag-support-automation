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


SUPPORTED_BACKENDS = ("memory", "chroma")
SUPPORTED_PIPELINES = ("native", "langchain")


def _load_chunks(settings) -> list:
    chunks = []
    try:
        for doc in load_documents("data/sample_docs"):
            chunks.extend(split_document(doc, settings.chunk_size, settings.chunk_overlap))
    except FileNotFoundError:
        pass
    return chunks


def build_pipeline(settings):
    """Build the pipeline the settings ask for. Offline components by default,
    so the service boots without API keys.

    ``PIPELINE``: ``native`` (hand-rolled ``RagChain``) or ``langchain`` (the
    same recipe in LCEL). ``VECTOR_BACKEND``: ``memory`` or ``chroma``. Both
    pipelines share the embedder, chunks, ``TOP_K``, ``MIN_SCORE``, prompts and
    escalation message, so they are directly comparable.
    """
    if settings.vector_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"VECTOR_BACKEND={settings.vector_backend!r} is not wired into the API; "
            f"use one of {SUPPORTED_BACKENDS}"
        )
    if settings.pipeline not in SUPPORTED_PIPELINES:
        raise ValueError(f"PIPELINE={settings.pipeline!r}; use one of {SUPPORTED_PIPELINES}")

    embedder = HashingEmbedder()
    chunks = _load_chunks(settings)

    if settings.pipeline == "langchain":
        from ..generation.langchain_pipeline import (
            LangChainEmbeddings,
            LangChainRagPipeline,
            build_vectorstore,
        )

        embeddings = LangChainEmbeddings(embedder)
        vectorstore = build_vectorstore(chunks, embeddings, settings.vector_backend)
        return LangChainRagPipeline(
            vectorstore,
            top_k=settings.top_k,
            min_score=settings.min_score,
            openai_api_key=settings.openai_api_key,
            chat_model=settings.chat_model,
        )

    if settings.vector_backend == "chroma":
        from ..vectorstore.chroma_store import ChromaStore

        store = ChromaStore()
    else:
        store = InMemoryStore()
    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        store.add(vectors, [{"text": c.text, "source": c.source} for c in chunks])

    retriever = Retriever(embedder, store, top_k=settings.top_k, min_score=settings.min_score)
    llm = None if settings.openai_api_key else _offline_llm
    return RagChain(retriever, llm=llm, chat_model=settings.chat_model)


@lru_cache(maxsize=1)
def get_chain():
    return build_pipeline(get_settings())


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
