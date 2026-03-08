"""
Data models for the AI Decision Support System.

Defines structured response objects including citations, risk indicators,
confidence scores, and sourced reasoning for transparent decision support.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Categorical risk levels for decision indicators."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskIndicator(BaseModel):
    """A single risk factor identified in the source documents."""
    category: str = Field(description="Category of the risk (e.g. 'Financial', 'Operational')")
    description: str = Field(description="Human-readable description of the risk")
    level: RiskLevel = Field(description="Severity level of the risk")
    score: float = Field(ge=0.0, le=1.0, description="Numeric risk score from 0 (none) to 1 (critical)")
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="IDs of document chunks where this risk was identified",
    )

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 4)


class Citation(BaseModel):
    """A source citation linking a claim back to its originating document chunk."""
    chunk_id: str = Field(description="Unique identifier of the source chunk")
    document_name: str = Field(description="Name/title of the source document")
    page_or_section: Optional[str] = Field(
        default=None,
        description="Page number or section reference within the document",
    )
    excerpt: str = Field(description="Verbatim text excerpt from the source chunk")
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="Semantic similarity score between query and this chunk",
    )

    @field_validator("relevance_score")
    @classmethod
    def round_relevance(cls, v: float) -> float:
        return round(v, 4)


class ContextWindowUsage(BaseModel):
    """Tracks how the LLM context window was used during generation."""
    total_capacity_tokens: int = Field(description="Total context window capacity in tokens")
    context_tokens_used: int = Field(description="Tokens consumed by retrieved context")
    prompt_tokens_used: int = Field(description="Tokens consumed by the prompt template")
    response_tokens_reserved: int = Field(description="Tokens reserved for the LLM response")
    utilization_pct: float = Field(ge=0.0, le=100.0, description="Percentage of context window used")
    chunks_included: int = Field(description="Number of context chunks included")
    chunks_truncated: int = Field(description="Number of chunks dropped due to token budget")


class DocumentChunk(BaseModel):
    """A single chunk of text extracted from a document, ready for embedding."""
    chunk_id: str = Field(description="Unique identifier for this chunk")
    document_name: str = Field(description="Source document name")
    text: str = Field(description="Text content of the chunk")
    metadata: dict = Field(default_factory=dict, description="Arbitrary metadata (page, section, etc.)")
    token_count: int = Field(default=0, description="Approximate token count of the chunk text")


class DecisionResponse(BaseModel):
    """
    Structured response from the RAG decision-support system.

    Contains a complete analysis including summary, risk indicators,
    confidence score, sourced citations, and reasoning transparency.
    """
    query: str = Field(description="The original user query")
    summary: str = Field(description="High-level summary of the decision-support analysis")
    answer: str = Field(description="Detailed answer to the query with inline citation markers [N]")
    risk_indicators: List[RiskIndicator] = Field(
        default_factory=list,
        description="Identified risks relevant to the query",
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the answer (0 = no evidence, 1 = definitive evidence)",
    )
    confidence_rationale: str = Field(
        description="Explanation of why this confidence score was assigned"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Source citations supporting the answer",
    )
    reasoning: str = Field(
        description="Step-by-step reasoning chain showing how the answer was derived"
    )
    hallucination_flags: List[str] = Field(
        default_factory=list,
        description="Warnings about potential hallucinations or low-confidence claims",
    )
    context_window_usage: Optional[ContextWindowUsage] = Field(
        default=None,
        description="Metrics on how the LLM context window was utilized",
    )

    @field_validator("confidence_score")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)

    def confidence_label(self) -> str:
        """Return a human-readable label for the confidence score."""
        if self.confidence_score >= 0.75:
            return "HIGH"
        if self.confidence_score >= 0.50:
            return "MEDIUM"
        return "LOW"

    def highest_risk_level(self) -> Optional[RiskLevel]:
        """Return the highest risk level found among all indicators."""
        if not self.risk_indicators:
            return None
        order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}
        return max(self.risk_indicators, key=lambda r: order[r.level]).level
