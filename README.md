# AI Decision Support System — RAG Portfolio Project

A production-quality **Retrieval-Augmented Generation (RAG)** system that acts as an AI decision-support analyst, delivering structured answers enriched with **citations**, **risk indicators**, **confidence scores**, and **sourced reasoning**.

Built as a portfolio project to demonstrate deep understanding of modern RAG architecture from ingestion to structured generation.

---

## What This Project Demonstrates

| Skill Area | Implementation |
|---|---|
| **Document Ingestion** | Multi-format loader (TXT, Markdown, PDF) with metadata extraction |
| **Chunking Strategies** | Fixed-size (token-aware with overlap), sentence-boundary, and semantic (embedding-based) |
| **Embeddings** | `sentence-transformers` with offline TF-IDF+SVD fallback |
| **Vector Databases** | ChromaDB persistent store with cosine-similarity ANN search |
| **Semantic Search** | Embedding-based retrieval with configurable top-k and minimum relevance threshold |
| **Context Retrieval** | Token-budget-aware context window management |
| **Structured Generation** | JSON-schema-enforced responses via OpenAI (with deterministic mock fallback) |
| **RAG Fundamentals** | Full pipeline: ingest → chunk → embed → store → retrieve → generate |
| **Prompt Templating** | Versioned system + user prompt templates with grounding and citation rules |
| **Context Window Management** | Token counting (tiktoken), budget allocation, utilization metrics |
| **Hallucination Mitigation** | Strict grounding, uncertainty instruction, confidence calibration, self-declared flags |

---

## Output Schema

Every query produces a `DecisionResponse` with:

```python
{
  "query":                 "What are the main financial risks?",
  "summary":               "2-3 sentence high-level summary ...",
  "answer":                "Detailed answer with inline [1][2] citations ...",
  "risk_indicators": [
    {
      "category":          "Financial",
      "description":       "Revenue concentration: top-3 clients = 62% of ARR",
      "level":             "high",      # low | medium | high | critical
      "score":             0.75,        # 0.0 - 1.0
      "source_chunk_ids":  ["doc-chunk-3-abc123"]
    }
  ],
  "confidence_score":      0.82,        # 0.0 - 1.0
  "confidence_rationale":  "...",
  "citations": [
    {
      "chunk_id":          "techcorp_risk_report_2024.txt-chunk-4-...",
      "document_name":     "techcorp_risk_report_2024.txt",
      "page_or_section":   null,
      "excerpt":           "TechCorp derives 62% of its total revenue ...",
      "relevance_score":   0.876
    }
  ],
  "reasoning":             "Step 1. ... Step 2. ...",
  "hallucination_flags":   [],
  "context_window_usage":  {
    "total_capacity_tokens":    3000,
    "context_tokens_used":      381,
    "prompt_tokens_used":       500,
    "response_tokens_reserved": 1000,
    "utilization_pct":          29.4,
    "chunks_included":          5,
    "chunks_truncated":         0
  }
}
```

---

## Project Structure

```
AI-Decision-Support-System-RAG/
├── config.py                  # Configuration (env vars with defaults)
├── main.py                    # CLI entry point (Typer + Rich)
├── requirements.txt
├── .env.example               # Environment variable template
│
├── src/
│   ├── __init__.py
│   ├── models.py              # Pydantic data models
│   ├── ingestion.py           # Document loading (TXT, MD, PDF)
│   ├── chunking.py            # Three chunking strategies
│   ├── embeddings.py          # Embeddings (sentence-transformers + TF-IDF fallback)
│   ├── vector_store.py        # ChromaDB vector store wrapper
│   ├── retrieval.py           # Semantic search + context window management
│   ├── generation.py          # Structured generation (OpenAI / mock)
│   └── prompts.py             # Prompt templates
│
├── tests/
│   ├── test_chunking.py
│   ├── test_generation.py
│   ├── test_ingestion.py
│   ├── test_models.py
│   └── test_prompts.py
│
└── sample_docs/
    ├── techcorp_risk_report_2024.txt   # Sample annual risk assessment
    └── investment_memo_techcorp.txt    # Sample investment memorandum
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

> Without an API key, the system uses a **deterministic mock generator** that still exercises the full retrieval pipeline and produces the correct response structure.

### 3. Run the demo

```bash
python main.py demo
```

This ingests the sample documents, runs three example queries, and displays structured output including risk indicators, citations, and confidence scores.

### 4. Ingest your own documents

```bash
# Single document
python main.py ingest /path/to/report.pdf

# Entire directory
python main.py ingest /path/to/docs/ --strategy sentence

# Available strategies: fixed_size | sentence | semantic
python main.py ingest /path/to/docs/ --strategy fixed_size --chunk-size 256
```

### 5. Query the system

```bash
# Interactive query
python main.py query "What are the main financial risks?"

# With options
python main.py query "Summarise key findings" --top-k 8 --min-score 0.4

# Raw JSON output
python main.py query "What is the risk exposure score?" --json

# Filter to a specific document
python main.py query "What is the IRR estimate?" --doc-filter "investment_memo_techcorp.txt"
```

### 6. Manage the store

```bash
python main.py status   # show indexed documents and chunk count
python main.py clear    # delete all stored data
```

---

## RAG Architecture Deep-Dive

### Chunking Strategies

| Strategy | How it works | Best for |
|---|---|---|
| `fixed_size` | Splits on token count with configurable overlap | Uniform, predictable chunk sizes |
| `sentence` | Groups N sentences respecting token ceiling | Preserving natural language boundaries |
| `semantic` | Starts new chunk when embedding similarity drops | Topic-aware splitting |

### Embedding Backend

1. **Primary**: `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional dense vectors, state-of-the-art semantic similarity.
2. **Offline fallback**: TF-IDF + Truncated SVD (LSA) — works without internet access or GPU. Used automatically if the primary model fails to load.

All embeddings are **L2-normalised** so cosine similarity = dot product (fast ANN search).

### Context Window Management

The retriever tracks token budgets using `tiktoken`:
- Allocates a fixed slice for the prompt template
- Reserves tokens for the LLM response
- Fits as many high-relevance chunks as possible in the remaining budget
- Reports exact utilisation in the response metadata

### Hallucination Mitigation

Four layers of defence are used:

1. **Strict grounding prompt**: LLM is instructed to use only the provided context.
2. **Uncertainty instruction**: LLM must say "evidence is insufficient" rather than guess.
3. **Confidence calibration**: LLM scores its own certainty (0.0-1.0) with a written rationale.
4. **Self-declared flags**: LLM explicitly lists any claims it is less than 80% confident about.

---

## Running Tests

```bash
pytest tests/ -v
```

63 tests covering: models, ingestion, chunking (3 strategies), prompt templates, and generation (mock mode).

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(none)* | OpenAI API key; if unset, mock mode is used |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model identifier |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |
