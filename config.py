"""
Configuration module for the AI Decision Support System.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# LLM settings
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding settings
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 output dimension

# Vector store settings
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")

# Chunking settings
DEFAULT_CHUNK_SIZE: int = 512          # tokens
DEFAULT_CHUNK_OVERLAP: int = 64        # tokens
SENTENCE_CHUNK_MAX_SENTENCES: int = 5  # sentences per chunk

# Retrieval settings
DEFAULT_TOP_K: int = 5                  # number of chunks to retrieve
MIN_RELEVANCE_SCORE: float = 0.3       # minimum cosine similarity score

# Context window management
MAX_CONTEXT_TOKENS: int = 3000         # max tokens for context passed to LLM
RESPONSE_RESERVE_TOKENS: int = 1000    # tokens reserved for LLM response

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD: float = 0.75
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.50

# Risk thresholds
RISK_SCORE_HIGH: float = 0.7
RISK_SCORE_MEDIUM: float = 0.4

# Sample docs directory
SAMPLE_DOCS_DIR: Path = Path(__file__).parent / "sample_docs"
