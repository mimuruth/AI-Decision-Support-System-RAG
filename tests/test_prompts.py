"""Tests for prompts module."""
from src.prompts import (
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    format_context_passages,
    build_rag_prompt,
)
from src.models import DocumentChunk


def _make_chunk(text: str, doc_name: str = "test.txt", chunk_id: str = "chunk-001") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_name=doc_name,
        text=text,
        token_count=len(text.split()),
    )


class TestSystemPrompt:
    def test_contains_grounding_instruction(self):
        assert "GROUNDING" in SYSTEM_PROMPT or "EXCLUSIVELY" in SYSTEM_PROMPT

    def test_contains_citation_instruction(self):
        assert "CITATIONS" in SYSTEM_PROMPT or "citation" in SYSTEM_PROMPT.lower()

    def test_contains_uncertainty_instruction(self):
        assert "UNCERTAINTY" in SYSTEM_PROMPT or "don't know" in SYSTEM_PROMPT.lower()


class TestFormatContextPassages:
    def test_empty_chunks(self):
        result = format_context_passages([])
        assert result == ""

    def test_single_chunk_numbered(self):
        chunk = _make_chunk("Test passage text.")
        result = format_context_passages([(chunk, 0.85)])
        assert "[1]" in result
        assert "Test passage text." in result
        assert "test.txt" in result

    def test_multiple_chunks_numbered_sequentially(self):
        chunks = [
            (_make_chunk("First passage.", "doc1.txt", "c1"), 0.9),
            (_make_chunk("Second passage.", "doc2.txt", "c2"), 0.75),
        ]
        result = format_context_passages(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "First passage." in result
        assert "Second passage." in result

    def test_relevance_score_included(self):
        chunk = _make_chunk("Content.")
        result = format_context_passages([(chunk, 0.876)])
        assert "0.876" in result

    def test_separator_between_passages(self):
        chunks = [
            (_make_chunk("A", chunk_id="c1"), 0.9),
            (_make_chunk("B", chunk_id="c2"), 0.8),
        ]
        result = format_context_passages(chunks)
        assert "---" in result


class TestBuildRagPrompt:
    def test_query_included_in_prompt(self):
        chunk = _make_chunk("Relevant info.")
        prompt = build_rag_prompt("What are the risks?", [(chunk, 0.8)])
        assert "What are the risks?" in prompt

    def test_context_included_in_prompt(self):
        chunk = _make_chunk("Important finding here.")
        prompt = build_rag_prompt("Query?", [(chunk, 0.7)])
        assert "Important finding here." in prompt

    def test_empty_context_builds_prompt(self):
        prompt = build_rag_prompt("Query with no context", [])
        assert "Query with no context" in prompt
