"""Tests for the generation module (mock mode only – no API key required)."""
import pytest

from src.generation import generate_response, _extract_json, _build_citations
from src.models import DocumentChunk, DecisionResponse, ContextWindowUsage


def _make_chunk(text: str = "Sample document text.") -> DocumentChunk:
    return DocumentChunk(
        chunk_id="test-chunk-001",
        document_name="test_doc.txt",
        text=text,
        token_count=len(text.split()),
    )


class TestExtractJson:
    def test_parses_clean_json(self):
        data = _extract_json('{"key": "value"}')
        assert data["key"] == "value"

    def test_extracts_json_from_noisy_text(self):
        text = 'Some prefix text\n{"key": "value"}\nsome suffix'
        data = _extract_json(text)
        assert data["key"] == "value"

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("not json at all ::::")


class TestBuildCitations:
    def test_empty_input(self):
        citations = _build_citations([])
        assert citations == []

    def test_builds_citation_from_chunk(self):
        chunk = _make_chunk("This is test content for citation.")
        citations = _build_citations([(chunk, 0.85)])
        assert len(citations) == 1
        assert citations[0].chunk_id == chunk.chunk_id
        assert citations[0].document_name == "test_doc.txt"
        assert citations[0].relevance_score == 0.85

    def test_excerpt_truncated_at_300_chars(self):
        long_text = "word " * 100
        chunk = _make_chunk(long_text)
        citations = _build_citations([(chunk, 0.9)])
        assert len(citations[0].excerpt) <= 305  # 300 + "..."


class TestGenerateResponseMockMode:
    """Tests for mock generation (no OpenAI API key needed)."""

    def test_returns_decision_response(self):
        chunk = _make_chunk("Revenue concentration poses a financial risk.")
        response = generate_response(
            "What are the financial risks?",
            [(chunk, 0.75)],
        )
        assert isinstance(response, DecisionResponse)

    def test_query_preserved(self):
        chunk = _make_chunk("Sample text.")
        response = generate_response("My test query", [(chunk, 0.8)])
        assert response.query == "My test query"

    def test_citations_populated(self):
        chunk = _make_chunk("Important information here.")
        response = generate_response("Question?", [(chunk, 0.72)])
        assert len(response.citations) == 1
        assert response.citations[0].document_name == "test_doc.txt"

    def test_confidence_positive_with_context(self):
        chunk = _make_chunk("Relevant evidence.")
        response = generate_response("Query?", [(chunk, 0.85)])
        assert response.confidence_score > 0.0

    def test_low_confidence_with_empty_context(self):
        response = generate_response("Obscure query with no context", [])
        assert response.confidence_score < 0.3

    def test_context_window_usage_passed_through(self):
        usage = ContextWindowUsage(
            total_capacity_tokens=4096,
            context_tokens_used=800,
            prompt_tokens_used=200,
            response_tokens_reserved=500,
            utilization_pct=24.4,
            chunks_included=3,
            chunks_truncated=0,
        )
        chunk = _make_chunk("Text.")
        response = generate_response("Q?", [(chunk, 0.5)], context_usage=usage)
        assert response.context_window_usage is not None
        assert response.context_window_usage.chunks_included == 3

    def test_mock_flag_in_hallucination_flags(self):
        response = generate_response("Q?", [])
        # Mock mode should include a flag indicating mock generation
        assert any("MOCK" in flag.upper() for flag in response.hallucination_flags)
