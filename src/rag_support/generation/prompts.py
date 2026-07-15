"""Prompt templates for grounded technical-support answers."""

SYSTEM_PROMPT = """\
You are a technical support assistant. Answer ONLY from the provided context.
If the context does not contain the answer, say so and suggest escalating to a
human engineer. Always cite the source document for each claim, e.g. [manual.md].
"""

ANSWER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer (with citations):"""


def format_context(results) -> str:
    return "\n\n".join(f"[{r.source}]\n{r.text}" for r in results)


def build_user_prompt(question: str, results) -> str:
    return ANSWER_TEMPLATE.format(context=format_context(results), question=question)
