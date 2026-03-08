"""
Document ingestion module.

Handles loading documents from disk in multiple formats (plain text,
Markdown, PDF) and normalising them into a consistent plain-text
representation ready for the chunking stage.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional PDF support – gracefully degrade if PyPDF2 is not installed.
try:
    import PyPDF2  # type: ignore
    _PDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDF_AVAILABLE = False
    logger.warning("PyPDF2 not installed – PDF ingestion will be unavailable.")

# Optional Markdown support.
try:
    import markdown  # type: ignore
    import re as _re
    _MD_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MD_AVAILABLE = False
    logger.warning("markdown package not installed – Markdown files will be read as plain text.")


class IngestedDocument:
    """Container for a document loaded from disk."""

    def __init__(self, name: str, text: str, metadata: Optional[Dict] = None) -> None:
        self.name = name
        self.text = text.strip()
        self.metadata: Dict = metadata or {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"IngestedDocument(name={self.name!r}, chars={len(self.text)})"


def _strip_html(html: str) -> str:
    """Remove HTML tags from a string (used after Markdown → HTML conversion)."""
    return _re.sub(r"<[^>]+>", "", html)


def load_text(path: Path) -> IngestedDocument:
    """Load a plain-text (.txt) file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return IngestedDocument(name=path.name, text=text, metadata={"source": str(path), "type": "txt"})


def load_markdown(path: Path) -> IngestedDocument:
    """Load a Markdown (.md) file, stripping HTML tags if markdown is available."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if _MD_AVAILABLE:
        html = markdown.markdown(raw)
        text = _strip_html(html)
    else:
        text = raw
    return IngestedDocument(name=path.name, text=text, metadata={"source": str(path), "type": "markdown"})


def load_pdf(path: Path) -> IngestedDocument:
    """Load a PDF file and extract text from all pages."""
    if not _PDF_AVAILABLE:
        raise RuntimeError("PyPDF2 is required to ingest PDF files. Run: pip install pypdf2")
    pages: List[str] = []
    with open(path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[Page {page_num}]\n{page_text}")
    text = "\n\n".join(pages)
    return IngestedDocument(
        name=path.name,
        text=text,
        metadata={"source": str(path), "type": "pdf", "page_count": len(reader.pages)},
    )


_LOADERS = {
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
}


def load_document(path: str | Path) -> IngestedDocument:
    """
    Load a single document from *path*.

    Supports .txt, .md / .markdown, and .pdf formats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported: {list(_LOADERS.keys())}"
        )
    doc = loader(path)
    logger.info("Loaded document '%s' (%d chars)", doc.name, len(doc.text))
    return doc


def load_directory(directory: str | Path, recursive: bool = False) -> List[IngestedDocument]:
    """
    Load all supported documents from a directory.

    Args:
        directory: Path to the directory.
        recursive: If True, also search sub-directories.

    Returns:
        List of :class:`IngestedDocument` objects (skips unsupported files).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    docs: List[IngestedDocument] = []
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            try:
                docs.append(load_document(path))
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to load '%s': %s", path, exc)
    logger.info("Loaded %d document(s) from '%s'", len(docs), directory)
    return docs
