"""Split documents into overlapping chunks suitable for embedding."""

from dataclasses import dataclass

from .loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    position: int


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Greedy paragraph-aware splitter with sliding-window overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
            continue
        if current:
            chunks.append(current)
        while len(para) > chunk_size:
            chunks.append(para[:chunk_size])
            para = para[chunk_size - overlap:]
        current = para
    if current:
        chunks.append(current)
    return chunks


def split_document(doc: Document, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    return [
        Chunk(text=t, source=doc.source, position=i)
        for i, t in enumerate(split_text(doc.text, chunk_size, overlap))
    ]
