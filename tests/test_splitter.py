import pytest

from rag_support.ingestion.loader import Document
from rag_support.ingestion.splitter import split_document, split_text


def test_short_text_single_chunk():
    assert split_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_paragraphs_grouped_within_chunk_size():
    text = "para one\n\npara two\n\npara three"
    chunks = split_text(text, chunk_size=1000, overlap=50)
    assert len(chunks) == 1


def test_long_paragraph_is_wrapped():
    text = "x" * 2500
    chunks = split_text(text, chunk_size=800, overlap=100)
    assert all(len(c) <= 800 for c in chunks)
    assert sum(len(c) for c in chunks) >= 2500


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_split_document_positions():
    doc = Document(text=("word " * 50 + "\n\n") * 20, source="manual.md", metadata={})
    chunks = split_document(doc, chunk_size=300, overlap=40)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "manual.md" for c in chunks)
