"""
Prompt templates for the AI Decision Support System.

Centralises all LLM prompts to make them easy to audit, version, and tune.
Templates follow the pattern: fill in variables, then pass to the LLM.

Hallucination mitigation strategies employed:
- Strict grounding instruction ("only use the provided context")
- Explicit uncertainty instruction ("say 'I don't know' if unsure")
- Citation requirement ("cite every factual claim with [N]")
- Confidence calibration ("rate your confidence 0–1")
- Self-check reasoning chain ("think step by step")
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert AI decision-support analyst. Your role is to analyse \
source documents and answer questions with precision, transparency, and \
rigorous citation. You MUST adhere to the following rules:

1. GROUNDING: Base your answer EXCLUSIVELY on the provided context passages. \
   Do NOT introduce external knowledge or assumptions beyond what is stated.
2. CITATIONS: Every factual claim MUST be followed by an inline citation \
   marker such as [1], [2], etc., matching the numbered sources listed at the end.
3. UNCERTAINTY: If the context does not contain enough information to answer \
   confidently, explicitly state "The available evidence is insufficient to \
   determine …" rather than speculating.
4. RISK: Identify and clearly label any risks, concerns, or uncertainties \
   mentioned in the context that are relevant to the query.
5. CONFIDENCE: Assign a numeric confidence score (0.0–1.0) reflecting how \
   well the context supports your answer. 0 = no relevant evidence found; \
   1.0 = conclusive evidence directly addresses the query.
6. REASONING: Provide a transparent, step-by-step reasoning chain so the \
   reader can verify your logic.
7. HALLUCINATION CHECK: At the end, list any claims you are less than 80% \
   certain about and flag them explicitly.
"""

# ---------------------------------------------------------------------------
# Main RAG prompt
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
## Query
{query}

## Context Passages
{context}

## Instructions
Using ONLY the context passages above, produce a structured analysis in \
valid JSON matching this exact schema:

{{
  "summary": "<2–3 sentence high-level summary>",
  "answer": "<detailed answer with inline [N] citations>",
  "risk_indicators": [
    {{
      "category": "<risk category>",
      "description": "<what the risk is>",
      "level": "<low|medium|high|critical>",
      "score": <0.0–1.0>,
      "source_chunk_ids": ["<chunk_id>", ...]
    }}
  ],
  "confidence_score": <0.0–1.0>,
  "confidence_rationale": "<why you assigned this score>",
  "reasoning": "<step-by-step reasoning chain>",
  "hallucination_flags": ["<claim you are unsure about>", ...]
}}

Rules:
- Cite every claim. Use [N] where N is the 1-based index of the source passage.
- If a risk is not present, return an empty list for risk_indicators.
- If nothing relevant is found, set confidence_score to 0.0 and explain in \
  confidence_rationale.
- Return ONLY the JSON object – no additional text before or after.
"""

# ---------------------------------------------------------------------------
# Context formatting helper
# ---------------------------------------------------------------------------

def format_context_passages(
    chunks_with_scores: list,  # List[Tuple[DocumentChunk, float]]
    include_metadata: bool = True,
) -> str:
    """
    Format retrieved chunks into a numbered list of context passages.

    Args:
        chunks_with_scores: List of ``(DocumentChunk, score)`` tuples.
        include_metadata: Whether to include source document name.

    Returns:
        Formatted string ready to be inserted into :data:`RAG_PROMPT_TEMPLATE`.
    """
    lines: list[str] = []
    for i, (chunk, score) in enumerate(chunks_with_scores, start=1):
        header = f"[{i}]"
        if include_metadata:
            header += f" Source: {chunk.document_name}"
            if chunk.metadata.get("page_or_section"):
                header += f" | {chunk.metadata['page_or_section']}"
            header += f" | chunk_id: {chunk.chunk_id} | relevance: {score:.3f}"
        lines.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n---\n\n".join(lines)


def build_rag_prompt(query: str, chunks_with_scores: list) -> str:
    """
    Build the full user-turn RAG prompt.

    Args:
        query: The original user question.
        chunks_with_scores: Retrieved ``(DocumentChunk, score)`` pairs.

    Returns:
        Formatted prompt string to pass as the user message.
    """
    context = format_context_passages(chunks_with_scores)
    return RAG_PROMPT_TEMPLATE.format(query=query, context=context)
