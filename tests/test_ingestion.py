"""Tests for document ingestion module."""
import tempfile
from pathlib import Path

import pytest

from src.ingestion import load_document, load_directory, IngestedDocument


def _write_temp(content: str, suffix: str) -> Path:
    """Write content to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


class TestLoadText:
    def test_loads_plain_text(self):
        path = _write_temp("Hello world. This is a test.", ".txt")
        doc = load_document(path)
        assert isinstance(doc, IngestedDocument)
        assert "Hello world" in doc.text
        assert doc.name == path.name
        assert doc.metadata["type"] == "txt"

    def test_strips_leading_trailing_whitespace(self):
        path = _write_temp("  \n  content  \n  ", ".txt")
        doc = load_document(path)
        assert doc.text == "content"


class TestLoadMarkdown:
    def test_loads_markdown_as_plain_text(self):
        path = _write_temp("# Heading\n\nSome **bold** text.", ".md")
        doc = load_document(path)
        assert "Heading" in doc.text
        assert doc.metadata["type"] == "markdown"

    def test_markdown_extension_alias(self):
        path = _write_temp("# Title\n\nParagraph.", ".markdown")
        doc = load_document(path)
        assert "Title" in doc.text


class TestLoadDirectory:
    def test_loads_all_supported_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("Alpha document.", encoding="utf-8")
        (tmp_path / "b.md").write_text("# Beta\n\nBeta document.", encoding="utf-8")
        (tmp_path / "c.ignore").write_text("ignored", encoding="utf-8")

        docs = load_directory(tmp_path)
        names = [d.name for d in docs]
        assert "a.txt" in names
        assert "b.md" in names
        assert len(docs) == 2  # c.ignore is skipped

    def test_empty_directory_returns_empty_list(self, tmp_path):
        docs = load_directory(tmp_path)
        assert docs == []

    def test_raises_for_missing_directory(self):
        with pytest.raises(NotADirectoryError):
            load_directory("/nonexistent/path/xyz")


class TestLoadDocumentErrors:
    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/file.txt")

    def test_raises_for_unsupported_extension(self, tmp_path):
        path = tmp_path / "file.xyz"
        path.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_document(path)
