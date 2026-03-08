"""
Structured generation module with citations and hallucination mitigation.

Sends the formatted RAG prompt to an LLM (OpenAI by default) and
parses the structured JSON response into a :class:`~src.models.DecisionResponse`.

Hallucination mitigation strategies:
- Response grounding via strict prompt instructions (see prompts.py)
- JSON schema enforcement (Pydantic validation catches schema violations)
- Confidence calibration (model scores its own certainty)
- Self-declared hallucination flags (model lists uncertain claims)
- Fallback mock mode when no API key is available (for testing/demo)
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Tuple

from src.models import (
    Citation,
    ContextWindowUsage,
    DecisionResponse,
    DocumentChunk,
    RiskIndicator,
    RiskLevel,
)
from src.prompts import SYSTEM_PROMPT, build_rag_prompt
from config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output, tolerating surrounding noise."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the JSON block with regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Citation builder
# ---------------------------------------------------------------------------

def _build_citations(
    chunks_with_scores: List[Tuple[DocumentChunk, float]],
) -> List[Citation]:
    """Build :class:`~src.models.Citation` objects from retrieved chunks."""
    citations: List[Citation] = []
    for chunk, score in chunks_with_scores:
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                page_or_section=chunk.metadata.get("page_or_section"),
                excerpt=chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
                relevance_score=score,
            )
        )
    return citations


# ---------------------------------------------------------------------------
# Mock generator (used when no OpenAI key is present)
# ---------------------------------------------------------------------------

def _mock_generate(
    query: str,
    chunks_with_scores: List[Tuple[DocumentChunk, float]],
    context_usage: Optional[ContextWindowUsage],
) -> DecisionResponse:
    """
    Return a deterministic mock response for demo/test purposes.

    This shows the full structure of a :class:`~src.models.DecisionResponse`
    without requiring an OpenAI API key.
    """
    citations = _build_citations(chunks_with_scores)
    has_context = bool(chunks_with_scores)
    confidence = round(min(0.5 + len(chunks_with_scores) * 0.05, 0.9), 2) if has_context else 0.1

    risk_indicators: List[RiskIndicator] = []
    if has_context:
        risk_indicators.append(
            RiskIndicator(
                category="Data Coverage",
                description="Answer is based on a limited number of retrieved passages.",
                level=RiskLevel.MEDIUM,
                score=0.45,
                source_chunk_ids=[c.chunk_id for c, _ in chunks_with_scores[:2]],
            )
        )

    answer_parts = []
    for i, (chunk, score) in enumerate(chunks_with_scores, start=1):
        snippet = chunk.text[:120].replace("\n", " ")
        answer_parts.append(f'According to the source material: "{snippet}..." [{i}]')

    answer = (
        " ".join(answer_parts)
        if answer_parts
        else "No relevant context was found to answer this query."
    )

    return DecisionResponse(
        query=query,
        summary=(
            f"Analysis of {len(chunks_with_scores)} retrieved passages for the query."
            if has_context
            else "No relevant documents were found for this query."
        ),
        answer=answer,
        risk_indicators=risk_indicators,
        confidence_score=confidence,
        confidence_rationale=(
            f"Confidence based on {len(chunks_with_scores)} retrieved passages with "
            f"an average relevance of "
            f"{sum(s for _, s in chunks_with_scores) / len(chunks_with_scores):.2f}."
            if has_context else "No relevant evidence found."
        ),
        citations=citations,
        reasoning=(
            "1. Embedded the query and searched the vector store.\n"
            f"2. Retrieved {len(chunks_with_scores)} chunks above the relevance threshold.\n"
            "3. Summarised key findings from retrieved passages.\n"
            "4. Assigned confidence score proportional to evidence coverage.\n"
            "NOTE: This is a MOCK response – set OPENAI_API_KEY for real generation."
        ),
        hallucination_flags=[
            "MOCK MODE: Response is deterministic and not LLM-generated."
        ],
        context_window_usage=context_usage,
    )


# ---------------------------------------------------------------------------
# OpenAI generator
# ---------------------------------------------------------------------------

def _openai_generate(
    query: str,
    chunks_with_scores: List[Tuple[DocumentChunk, float]],
    context_usage: Optional[ContextWindowUsage],
    model: str = OPENAI_MODEL,
) -> DecisionResponse:
    """Call the OpenAI Chat Completions API and parse the structured response."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required for LLM generation. Run: pip install openai"
        ) from exc

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_rag_prompt(query, chunks_with_scores)

    logger.info("Calling OpenAI model '%s' …", model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,   # deterministic – reduces hallucinations
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content or "{}"
    logger.debug("Raw LLM response: %s", raw_content[:500])

    data = _extract_json(raw_content)

    # Parse risk indicators
    raw_risks = data.get("risk_indicators", [])
    risk_indicators: List[RiskIndicator] = []
    for r in raw_risks:
        try:
            risk_indicators.append(
                RiskIndicator(
                    category=r.get("category", "General"),
                    description=r.get("description", ""),
                    level=RiskLevel(r.get("level", "medium").lower()),
                    score=float(r.get("score", 0.5)),
                    source_chunk_ids=r.get("source_chunk_ids", []),
                )
            )
        except Exception as exc:
            logger.warning("Skipped malformed risk indicator: %s – %s", r, exc)

    citations = _build_citations(chunks_with_scores)

    return DecisionResponse(
        query=query,
        summary=data.get("summary", ""),
        answer=data.get("answer", ""),
        risk_indicators=risk_indicators,
        confidence_score=float(data.get("confidence_score", 0.0)),
        confidence_rationale=data.get("confidence_rationale", ""),
        citations=citations,
        reasoning=data.get("reasoning", ""),
        hallucination_flags=data.get("hallucination_flags", []),
        context_window_usage=context_usage,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_response(
    query: str,
    chunks_with_scores: List[Tuple[DocumentChunk, float]],
    context_usage: Optional[ContextWindowUsage] = None,
    model: Optional[str] = None,
) -> DecisionResponse:
    """
    Generate a structured :class:`~src.models.DecisionResponse` for *query*.

    Automatically selects real (OpenAI) or mock generation depending on
    whether :envvar:`OPENAI_API_KEY` is configured.

    Args:
        query: The user's question.
        chunks_with_scores: Retrieved ``(chunk, similarity)`` pairs from the
            context retrieval stage.
        context_usage: Optional context window usage statistics.
        model: OpenAI model name override.

    Returns:
        A fully populated :class:`~src.models.DecisionResponse`.
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
        logger.info("No OpenAI API key detected – using mock generator.")
        return _mock_generate(query, chunks_with_scores, context_usage)

    return _openai_generate(
        query,
        chunks_with_scores,
        context_usage,
        model=model or OPENAI_MODEL,
    )
