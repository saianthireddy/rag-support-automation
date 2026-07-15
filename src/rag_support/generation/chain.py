"""End-to-end RAG chain: retrieve -> assemble prompt -> generate."""

from dataclasses import dataclass

from ..retrieval.retriever import Retriever
from .prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]
    context_used: int


class RagChain:
    def __init__(self, retriever: Retriever, llm=None, chat_model: str = "gpt-4o-mini"):
        """`llm` is any callable (system, user) -> str. Defaults to OpenAI."""
        self._retriever = retriever
        self._chat_model = chat_model
        self._llm = llm or self._openai_llm

    def _openai_llm(self, system: str, user: str) -> str:
        from openai import OpenAI  # lazy import

        response = OpenAI().chat.completions.create(
            model=self._chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def ask(self, question: str) -> RagAnswer:
        results = self._retriever.retrieve(question)
        if not results:
            return RagAnswer(
                answer=(
                    "I could not find relevant documentation. "
                    "Please escalate to a human engineer."
                ),
                sources=[],
                context_used=0,
            )
        answer = self._llm(SYSTEM_PROMPT, build_user_prompt(question, results))
        return RagAnswer(
            answer=answer,
            sources=sorted({r.source for r in results}),
            context_used=len(results),
        )
