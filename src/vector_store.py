"""
Vector store module backed by ChromaDB.

Provides a thin wrapper around ChromaDB for persisting document chunk
embeddings and running approximate-nearest-neighbour (ANN) searches.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import DocumentChunk
from config import CHROMA_DB_PATH, COLLECTION_NAME, DEFAULT_TOP_K

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Persistent vector store for document chunks using ChromaDB.

    Each document chunk is stored with its embedding, text, and metadata.
    Similarity search returns the top-k most relevant chunks for a query.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = CHROMA_DB_PATH,
    ) -> None:
        """
        Initialise (or reconnect to) a ChromaDB collection.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Local directory where ChromaDB persists data.
        """
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for the vector store. Run: pip install chromadb"
            ) from exc

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready – collection '%s' (%d items)",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """
        Add document chunks and their embeddings to the store.

        Args:
            chunks: List of :class:`~src.models.DocumentChunk` objects.
            embeddings: Embedding matrix of shape ``(len(chunks), dim)``.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_name": c.document_name,
                "token_count": c.token_count,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]
        embedding_list = embeddings.tolist()

        # ChromaDB upserts by id, so re-ingesting is idempotent
        self._collection.upsert(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Stored %d chunks in vector store.", len(chunks))

    def delete_document(self, document_name: str) -> int:
        """
        Remove all chunks belonging to *document_name*.

        Returns:
            Number of chunks deleted.
        """
        results = self._collection.get(where={"document_name": document_name})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Deleted %d chunks for document '%s'.", len(ids), document_name)
        return len(ids)

    def clear(self) -> None:
        """Delete **all** chunks from the collection."""
        results = self._collection.get()
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Cleared %d chunks from vector store.", len(ids))

    # ------------------------------------------------------------------
    # Read / query operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        where: Optional[Dict] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Find the *top_k* most similar chunks to *query_embedding*.

        Args:
            query_embedding: 1-D embedding vector for the query.
            top_k: Maximum number of results to return.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of ``(DocumentChunk, similarity_score)`` tuples sorted by
            descending similarity (best match first).
        """
        query_vec = query_embedding.tolist()
        kwargs: Dict = {
            "query_embeddings": [query_vec],
            "n_results": min(top_k, max(self._collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        ids: List[str] = results["ids"][0]
        documents: List[str] = results["documents"][0]
        metadatas: List[Dict] = results["metadatas"][0]
        # ChromaDB returns L2 or cosine distances; with cosine space,
        # distance = 1 - similarity, so similarity = 1 - distance.
        distances: List[float] = results["distances"][0]

        output: List[Tuple[DocumentChunk, float]] = []
        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, 1.0 - dist)
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_name=meta.get("document_name", "unknown"),
                text=doc_text,
                metadata={k: v for k, v in meta.items() if k not in ("document_name", "token_count")},
                token_count=int(meta.get("token_count", 0)),
            )
            output.append((chunk, round(similarity, 4)))

        return output

    def count(self) -> int:
        """Return the total number of chunks stored."""
        return self._collection.count()

    def list_documents(self) -> List[str]:
        """Return a deduplicated list of document names in the store."""
        results = self._collection.get(include=["metadatas"])
        names = {m.get("document_name", "") for m in results.get("metadatas", [])}
        return sorted(names - {""})
