"""
Chunking strategies for splitting documents into retrievable pieces.

Three strategies are provided:
- ``fixed_size``    – splits on approximate token counts (fast, consistent chunk sizes)
- ``sentence``      – groups consecutive sentences (preserves semantic boundaries)
- ``semantic``      – merges adjacent sentences until a similarity drop is detected

All strategies return a list of :class:`~src.models.DocumentChunk` objects.
"""
from __future__ import annotations

import re
import uuid
from typing import List

from src.models import DocumentChunk
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SENTENCE_CHUNK_MAX_SENTENCES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx_token_count(text: str) -> int:
    """Approximate token count using whitespace splitting (≈ GPT tokenisation)."""
    return len(text.split())


def _split_sentences(text: str) -> List[str]:
    """Split text into individual sentences."""
    # Handle common abbreviations to avoid false splits
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g)\.", r"\1<PERIOD>", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.replace("<PERIOD>", ".").strip() for s in sentences if s.strip()]


def _make_chunk(
    text: str,
    document_name: str,
    index: int,
    metadata: dict | None = None,
) -> DocumentChunk:
    chunk_id = f"{document_name}-chunk-{index}-{uuid.uuid4().hex[:8]}"
    return DocumentChunk(
        chunk_id=chunk_id,
        document_name=document_name,
        text=text,
        metadata=metadata or {},
        token_count=_approx_token_count(text),
    )


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunking
# ---------------------------------------------------------------------------

def fixed_size_chunking(
    text: str,
    document_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: dict | None = None,
) -> List[DocumentChunk]:
    """
    Split *text* into overlapping fixed-size chunks (measured in tokens).

    The overlap creates a sliding-window effect so context is not lost at
    chunk boundaries – an important RAG best practice.

    Args:
        text: Document text to chunk.
        document_name: Source document identifier.
        chunk_size: Maximum tokens per chunk.
        overlap: Number of tokens shared between consecutive chunks.
        metadata: Optional metadata to attach to every chunk.

    Returns:
        List of :class:`~src.models.DocumentChunk`.
    """
    words = text.split()
    chunks: List[DocumentChunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(_make_chunk(chunk_text, document_name, index, metadata))
        index += 1
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-based chunking
# ---------------------------------------------------------------------------

def sentence_chunking(
    text: str,
    document_name: str,
    max_sentences: int = SENTENCE_CHUNK_MAX_SENTENCES,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    metadata: dict | None = None,
) -> List[DocumentChunk]:
    """
    Group consecutive sentences into chunks respecting *max_sentences* and
    *max_tokens* limits.

    This strategy preserves sentence boundaries, making chunks more natural
    for the LLM to read.

    Args:
        text: Document text to chunk.
        document_name: Source document identifier.
        max_sentences: Maximum sentences per chunk.
        max_tokens: Token ceiling per chunk.
        metadata: Optional metadata to attach to every chunk.

    Returns:
        List of :class:`~src.models.DocumentChunk`.
    """
    sentences = _split_sentences(text)
    chunks: List[DocumentChunk] = []
    current: List[str] = []
    current_tokens = 0
    index = 0

    for sentence in sentences:
        sentence_tokens = _approx_token_count(sentence)
        if (
            current
            and (
                len(current) >= max_sentences
                or current_tokens + sentence_tokens > max_tokens
            )
        ):
            chunks.append(_make_chunk(" ".join(current), document_name, index, metadata))
            index += 1
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(_make_chunk(" ".join(current), document_name, index, metadata))

    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Semantic chunking
# ---------------------------------------------------------------------------

def semantic_chunking(
    text: str,
    document_name: str,
    similarity_threshold: float = 0.75,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    metadata: dict | None = None,
) -> List[DocumentChunk]:
    """
    Split text by detecting semantic topic changes between consecutive sentences.

    Uses cosine similarity of sentence embeddings to detect when the topic
    shifts, starting a new chunk at that boundary.  Falls back to
    :func:`sentence_chunking` if ``sentence-transformers`` is not available.

    Args:
        text: Document text to chunk.
        document_name: Source document identifier.
        similarity_threshold: Cosine similarity below which a new chunk begins.
        max_tokens: Hard token ceiling per chunk (prevents runaway chunks).
        metadata: Optional metadata to attach to every chunk.

    Returns:
        List of :class:`~src.models.DocumentChunk`.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        # Graceful fallback: sentence_transformers not available
        return sentence_chunking(text, document_name, metadata=metadata)

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [_make_chunk(text, document_name, 0, metadata)]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks: List[DocumentChunk] = []
    current: List[str] = []
    current_tokens = 0
    index = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = _approx_token_count(sentence)

        if current:
            # Compare embedding of this sentence with the last sentence in the chunk
            sim = float(
                cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[i - 1].reshape(1, -1),
                )[0][0]
            )
            boundary = sim < similarity_threshold or current_tokens + sentence_tokens > max_tokens
            if boundary:
                chunks.append(_make_chunk(" ".join(current), document_name, index, metadata))
                index += 1
                current = []
                current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(_make_chunk(" ".join(current), document_name, index, metadata))

    return chunks


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

STRATEGIES = {
    "fixed_size": fixed_size_chunking,
    "sentence": sentence_chunking,
    "semantic": semantic_chunking,
}


def chunk_document(
    text: str,
    document_name: str,
    strategy: str = "sentence",
    metadata: dict | None = None,
    **kwargs,
) -> List[DocumentChunk]:
    """
    Chunk a document using the named *strategy*.

    Args:
        text: Raw document text.
        document_name: Identifier for the source document.
        strategy: One of ``"fixed_size"``, ``"sentence"``, or ``"semantic"``.
        metadata: Optional metadata forwarded to every chunk.
        **kwargs: Additional keyword arguments passed to the strategy function.

    Returns:
        List of :class:`~src.models.DocumentChunk`.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from: {list(STRATEGIES)}")
    return STRATEGIES[strategy](text, document_name, metadata=metadata, **kwargs)
