"""
Embeddings module.

Produces dense vector embeddings for document chunks and queries.

Primary backend: ``sentence-transformers`` (requires model download).
Fallback backend: TF-IDF with SVD (works fully offline, no model download
needed) – used automatically when the sentence-transformers model cannot be
loaded (e.g. in offline/sandboxed environments).

All embeddings are L2-normalised so that cosine similarity reduces to a
dot product.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

_model_cache: dict = {}
_tfidf_state: dict = {}  # stores fitted TfidfVectorizer + TruncatedSVD


# ---------------------------------------------------------------------------
# Sentence-transformers backend
# ---------------------------------------------------------------------------

def _get_st_model(model_name: str):
    """Return (and cache) a SentenceTransformer model, or None if unavailable."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading embedding model '%s' ...", model_name)
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        return model
    except Exception as exc:
        logger.warning(
            "sentence-transformers model '%s' could not be loaded (%s). "
            "Falling back to TF-IDF embeddings.",
            model_name, exc,
        )
        _model_cache[model_name] = None
        return None


def _embed_with_st(texts: Sequence[str], model_name: str, batch_size: int) -> np.ndarray | None:
    model = _get_st_model(model_name)
    if model is None:
        return None
    embeddings: np.ndarray = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# TF-IDF + SVD offline fallback
# ---------------------------------------------------------------------------

def _embed_with_tfidf(
    texts: Sequence[str],
    dim: int = EMBEDDING_DIMENSION,
    fit: bool = False,
) -> np.ndarray:
    """
    Produce dense embeddings via TF-IDF + Truncated SVD (LSA).

    On the first call (or when ``fit=True``) the vectorizer and SVD are
    fitted on the supplied texts; subsequent calls use the cached state.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.decomposition import TruncatedSVD  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore

    texts_list = list(texts)
    actual_dim = min(dim, len(texts_list) - 1) if len(texts_list) > 1 else 1

    if fit or "pipeline" not in _tfidf_state:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8192, sublinear_tf=True)),
            ("svd", TruncatedSVD(n_components=actual_dim, random_state=42)),
        ])
        matrix = pipeline.fit_transform(texts_list).astype(np.float32)
        _tfidf_state["pipeline"] = pipeline
        _tfidf_state["dim"] = actual_dim
    else:
        pipeline = _tfidf_state["pipeline"]
        matrix = pipeline.transform(texts_list).astype(np.float32)

    # Pad or truncate to match EMBEDDING_DIMENSION
    if matrix.shape[1] < dim:
        pad = np.zeros((matrix.shape[0], dim - matrix.shape[1]), dtype=np.float32)
        matrix = np.hstack([matrix, pad])
    elif matrix.shape[1] > dim:
        matrix = matrix[:, :dim]

    return matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(
    texts: Sequence[str],
    model_name: str = EMBEDDING_MODEL,
    normalize: bool = True,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Produce dense embeddings for a list of texts.

    Tries sentence-transformers first; falls back to TF-IDF+SVD if the
    model is unavailable (e.g. no internet access).

    Args:
        texts: Sequence of strings to embed.
        model_name: Sentence-Transformers model identifier.
        normalize: If True, L2-normalise each embedding vector.
        batch_size: Batch size for sentence-transformers encoding.

    Returns:
        NumPy array of shape ``(len(texts), embedding_dim)``.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    embeddings = _embed_with_st(texts, model_name, batch_size)
    if embeddings is None:
        logger.info("Using TF-IDF fallback embeddings.")
        embeddings = _embed_with_tfidf(texts, dim=EMBEDDING_DIMENSION, fit=True)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

    return embeddings.astype(np.float32)


def embed_query(query: str, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Produce a single embedding for a query string.

    Returns a 1-D NumPy array of shape ``(embedding_dim,)``.
    """
    return embed_texts([query], model_name=model_name)[0]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two 1-D vectors.

    Assumes both are already L2-normalised (dot product == cosine similarity).
    """
    return float(np.dot(vec_a, vec_b))

