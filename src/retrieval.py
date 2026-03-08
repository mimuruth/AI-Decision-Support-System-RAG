"""
Context retrieval module.

Combines embedding-based semantic search with context window management
to retrieve the most relevant document chunks for a given query while
staying within the LLM's token budget.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from src.models import DocumentChunk, ContextWindowUsage
from src.embeddings import embed_query
from src.vector_store import VectorStore
from config import (
    DEFAULT_TOP_K,
    MIN_RELEVANCE_SCORE,
    MAX_CONTEXT_TOKENS,
    RESPONSE_RESERVE_TOKENS,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


def _count_tokens(text: str) -> int:
    """
    Estimate token count for the given text.

    Tries tiktoken for accurate GPT tokenisation; falls back to word
    count (a reasonable approximation).
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


class ContextRetriever:
    """
    Retrieves semantically relevant document chunks and manages context budget.

    This class is the core of the RAG pipeline:
    1. Embed the user query.
    2. Search the vector store for similar chunks.
    3. Filter by minimum relevance score.
    4. Fit as many chunks as possible within the context token budget.
    5. Return the selected chunks with usage statistics.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_RELEVANCE_SCORE,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> None:
        """
        Args:
            vector_store: Populated :class:`~src.vector_store.VectorStore`.
            model_name: Embedding model used to encode the query.
            top_k: Maximum number of chunks to retrieve before filtering.
            min_score: Minimum cosine similarity to include a chunk.
            max_context_tokens: Token budget for retrieved context.
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_tokens = max_context_tokens

    def retrieve(
        self,
        query: str,
        document_filter: Optional[str] = None,
        prompt_overhead_tokens: int = 500,
    ) -> Tuple[List[Tuple[DocumentChunk, float]], ContextWindowUsage]:
        """
        Retrieve relevant chunks for *query* within the token budget.

        Args:
            query: Natural-language query string.
            document_filter: If given, restrict search to this document name.
            prompt_overhead_tokens: Tokens consumed by the prompt template
                (system prompt + instructions, excluding context).

        Returns:
            Tuple of:
              - List of ``(DocumentChunk, similarity_score)`` sorted best-first.
              - :class:`~src.models.ContextWindowUsage` statistics.
        """
        # 1. Embed the query
        query_vec = embed_query(query, model_name=self.model_name)

        # 2. Search vector store
        where = {"document_name": document_filter} if document_filter else None
        candidates = self.vector_store.query(
            query_embedding=query_vec,
            top_k=self.top_k,
            where=where,
        )

        # 3. Filter by minimum relevance score
        candidates = [(chunk, score) for chunk, score in candidates if score >= self.min_score]

        if not candidates:
            logger.warning("No chunks met the minimum relevance threshold of %.2f.", self.min_score)

        # 4. Context window management – fit chunks within token budget
        available_tokens = self.max_context_tokens - prompt_overhead_tokens - RESPONSE_RESERVE_TOKENS
        selected: List[Tuple[DocumentChunk, float]] = []
        total_context_tokens = 0
        truncated = 0

        for chunk, score in candidates:
            chunk_tokens = _count_tokens(chunk.text)
            if total_context_tokens + chunk_tokens <= available_tokens:
                selected.append((chunk, score))
                total_context_tokens += chunk_tokens
            else:
                truncated += 1
                logger.debug(
                    "Chunk '%s' dropped – would exceed token budget (%d + %d > %d).",
                    chunk.chunk_id, total_context_tokens, chunk_tokens, available_tokens,
                )

        total_tokens_used = prompt_overhead_tokens + total_context_tokens
        total_capacity = self.max_context_tokens
        utilization = round(total_tokens_used / total_capacity * 100, 1) if total_capacity else 0.0

        usage = ContextWindowUsage(
            total_capacity_tokens=total_capacity,
            context_tokens_used=total_context_tokens,
            prompt_tokens_used=prompt_overhead_tokens,
            response_tokens_reserved=RESPONSE_RESERVE_TOKENS,
            utilization_pct=utilization,
            chunks_included=len(selected),
            chunks_truncated=truncated,
        )

        logger.info(
            "Retrieved %d chunks (%.1f%% context window used, %d truncated).",
            len(selected), utilization, truncated,
        )

        return selected, usage
