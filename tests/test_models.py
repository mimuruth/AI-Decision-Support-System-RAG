"""Tests for the data models."""
import pytest
from pydantic import ValidationError

from src.models import (
    RiskLevel,
    RiskIndicator,
    Citation,
    ContextWindowUsage,
    DocumentChunk,
    DecisionResponse,
)


class TestRiskIndicator:
    def test_valid_creation(self):
        r = RiskIndicator(
            category="Financial",
            description="Revenue concentration risk",
            level=RiskLevel.HIGH,
            score=0.75,
        )
        assert r.level == RiskLevel.HIGH
        assert r.score == 0.75

    def test_score_rounded(self):
        r = RiskIndicator(
            category="Ops",
            description="desc",
            level=RiskLevel.LOW,
            score=0.123456789,
        )
        assert r.score == 0.1235

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            RiskIndicator(
                category="X",
                description="d",
                level=RiskLevel.LOW,
                score=1.5,
            )


class TestCitation:
    def test_valid_creation(self):
        c = Citation(
            chunk_id="doc-chunk-1-abc",
            document_name="report.txt",
            excerpt="Some text excerpt.",
            relevance_score=0.88,
        )
        assert c.relevance_score == 0.88
        assert c.page_or_section is None

    def test_relevance_score_rounded(self):
        c = Citation(
            chunk_id="id",
            document_name="doc",
            excerpt="text",
            relevance_score=0.8765432,
        )
        assert c.relevance_score == 0.8765


class TestDecisionResponse:
    def _make_response(self, confidence=0.8) -> DecisionResponse:
        return DecisionResponse(
            query="What are the risks?",
            summary="Summary text.",
            answer="Answer text [1].",
            confidence_score=confidence,
            confidence_rationale="High relevance evidence found.",
            reasoning="Step 1. Step 2.",
        )

    def test_confidence_label_high(self):
        resp = self._make_response(confidence=0.8)
        assert resp.confidence_label() == "HIGH"

    def test_confidence_label_medium(self):
        resp = self._make_response(confidence=0.6)
        assert resp.confidence_label() == "MEDIUM"

    def test_confidence_label_low(self):
        resp = self._make_response(confidence=0.3)
        assert resp.confidence_label() == "LOW"

    def test_highest_risk_level_empty(self):
        resp = self._make_response()
        assert resp.highest_risk_level() is None

    def test_highest_risk_level(self):
        resp = self._make_response()
        resp.risk_indicators = [
            RiskIndicator(category="A", description="d", level=RiskLevel.LOW, score=0.2),
            RiskIndicator(category="B", description="d", level=RiskLevel.HIGH, score=0.7),
            RiskIndicator(category="C", description="d", level=RiskLevel.MEDIUM, score=0.5),
        ]
        assert resp.highest_risk_level() == RiskLevel.HIGH

    def test_confidence_score_rounded(self):
        resp = self._make_response(confidence=0.123456)
        assert resp.confidence_score == 0.1235

    def test_invalid_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            self._make_response(confidence=1.5)


class TestContextWindowUsage:
    def test_valid_creation(self):
        u = ContextWindowUsage(
            total_capacity_tokens=4096,
            context_tokens_used=1200,
            prompt_tokens_used=300,
            response_tokens_reserved=500,
            utilization_pct=36.6,
            chunks_included=5,
            chunks_truncated=0,
        )
        assert u.chunks_included == 5
        assert u.utilization_pct == 36.6
