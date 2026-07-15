"""Load technical manuals, SOPs and support docs from disk."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}


@dataclass
class Document:
    text: str
    source: str
    metadata: dict


def load_documents(root: str | Path) -> Iterator[Document]:
    """Yield one Document per supported file under *root* (recursive)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Document root not found: {root}")
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
            yield Document(
                text=path.read_text(encoding="utf-8", errors="ignore"),
                source=str(path.relative_to(root)),
                metadata={"suffix": path.suffix.lower()},
            )
