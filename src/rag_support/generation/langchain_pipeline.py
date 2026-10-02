"""The same RAG pipeline expressed in LangChain (LCEL).

This exists beside the hand-rolled ``RagChain`` rather than replacing it, so
the two can be compared on the same eval. Both use the same embedder, chunks,
relevance floor, prompts and escalation message, so any difference in the
numbers comes from the framework, not from a changed recipe.

    question ─┬─> retriever (k, score_threshold=MIN_SCORE) ─> docs
              └─> passthrough ─────────────────────────────> question
                        │
                        ▼
      no docs? ── yes ─> escalation message (no LLM call)
                        │ no
                        ▼
      prompt (SYSTEM_PROMPT + ANSWER_TEMPLATE) ─> LLM ─> str

Offline by default: with no ``OPENAI_API_KEY`` the "LLM" is a runnable that
returns the retrieved context, exactly like the native pipeline's fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from ..embeddings.embedder import Embedder
from ..ingestion.splitter import Chunk
from .chain import ESCALATION, RagAnswer
from .prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT


class _DropEmptyRetrievalWarning(logging.Filter):
    """An empty retrieval is the designed path to escalation, not a problem;
    LangChain logs it as a warning on every off-topic question."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("No relevant docs were retrieved")


logging.getLogger("langchain_core.vectorstores.base").addFilter(_DropEmptyRetrievalWarning())

OFFLINE_PREFIX = "OPENAI_API_KEY is not configured — returning retrieved context only.\n\n"


class LangChainEmbeddings(Embeddings):
    """Adapts any ``rag_support`` embedder to LangChain's ``Embeddings``."""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0]


class CosineInMemoryVectorStore(InMemoryVectorStore):
    """``InMemoryVectorStore`` already scores by cosine similarity but does not
    declare a relevance function, so ``similarity_score_threshold`` retrieval
    raises. Its scores ARE the relevance, so pass them through unchanged."""

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return lambda score: score

    def similarity_search_with_score_by_vector(self, embedding, k=4, filter=None, **kwargs):
        # A question made only of stopwords ("what is it?") embeds to the zero
        # vector, and LangChain's cosine raises on the resulting NaN. It has no
        # content to match, so it matches nothing: the chain escalates. (Chroma
        # and the native stores already score it 0, below any floor.)
        if not any(embedding):
            return []
        return super().similarity_search_with_score_by_vector(
            embedding, k=k, filter=filter, **kwargs
        )


def build_vectorstore(
    chunks: Sequence[Chunk], embeddings: Embeddings, backend: str = "memory"
) -> VectorStore:
    """Index chunks in LangChain's in-memory store or in Chroma (cosine)."""
    texts = [c.text for c in chunks]
    metadatas = [{"source": c.source} for c in chunks]
    if backend == "chroma":
        import uuid

        from chromadb.config import Settings
        from langchain_chroma import Chroma

        store = Chroma(
            collection_name=f"support-kb-lc-{uuid.uuid4().hex[:8]}",
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=Settings(anonymized_telemetry=False),
        )
    elif backend == "memory":
        store = CosineInMemoryVectorStore(embedding=embeddings)
    else:
        raise ValueError(f"Unknown vector store backend {backend!r}; use 'memory' or 'chroma'")
    if texts:
        store.add_texts(texts, metadatas=metadatas)
    return store


def _offline_llm(prompt_value) -> str:
    # Mirrors the native pipeline: with no model configured, hand back the
    # grounded context so the service still returns something useful.
    return OFFLINE_PREFIX + prompt_value.to_messages()[-1].content


def _default_llm(openai_api_key: str, chat_model: str) -> Runnable:
    if not openai_api_key:
        return RunnableLambda(_offline_llm)
    from langchain_openai import ChatOpenAI  # optional dependency, lazy import

    return ChatOpenAI(model=chat_model, temperature=0.1, api_key=openai_api_key)


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('source', '')}]\n{d.page_content}" for d in docs)


class LangChainRagPipeline:
    """Same contract as ``RagChain``: ``ask(question) -> RagAnswer``."""

    def __init__(
        self,
        vectorstore: VectorStore,
        top_k: int = 4,
        min_score: float = 0.0,
        llm: Runnable | Callable | None = None,
        openai_api_key: str = "",
        chat_model: str = "gpt-4o-mini",
    ):
        self.retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": min_score},
        )
        if llm is None:
            llm = _default_llm(openai_api_key, chat_model)
        elif not isinstance(llm, Runnable):
            llm = RunnableLambda(llm)

        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", ANSWER_TEMPLATE)]
        )
        generate = prompt | llm | StrOutputParser()

        def _answer(inputs: dict) -> RagAnswer:
            docs: list[Document] = inputs["docs"]
            if not docs:
                return RagAnswer(answer=ESCALATION, sources=[], context_used=0)
            text = generate.invoke(
                {"context": _format_context(docs), "question": inputs["question"]}
            )
            return RagAnswer(
                answer=text,
                sources=sorted({d.metadata.get("source", "") for d in docs}),
                context_used=len(docs),
            )

        self.chain: Runnable = RunnableParallel(
            docs=self.retriever, question=RunnablePassthrough()
        ) | RunnableLambda(_answer)

    def retrieve(self, question: str) -> list[Document]:
        if not question.strip():
            return []
        return self.retriever.invoke(question)

    def ask(self, question: str) -> RagAnswer:
        if not question.strip():
            return RagAnswer(answer=ESCALATION, sources=[], context_used=0)
        return self.chain.invoke(question)
