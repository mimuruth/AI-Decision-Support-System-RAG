"""
AI Decision Support System – Command Line Interface

A portfolio demonstration of Retrieval-Augmented Generation (RAG) that
provides structured decision support with:
  • Summaries
  • Risk indicators
  • Confidence scores
  • Sourced reasoning

Usage examples:
  python main.py ingest sample_docs/
  python main.py query "What are the main financial risks?"
  python main.py query "Summarise the key findings" --top-k 8
  python main.py status
  python main.py demo
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    MIN_RELEVANCE_SCORE,
)
from src.ingestion import load_document, load_directory
from src.chunking import chunk_document
from src.embeddings import embed_texts
from src.vector_store import VectorStore
from src.retrieval import ContextRetriever
from src.generation import generate_response
from src.models import RiskLevel

app = typer.Typer(
    name="rag-dss",
    help="AI Decision Support System powered by RAG",
    add_completion=False,
)
console = Console()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _get_store() -> VectorStore:
    return VectorStore(collection_name=COLLECTION_NAME, persist_directory=CHROMA_DB_PATH)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to a document file or directory"),
    strategy: str = typer.Option("sentence", help="Chunking strategy: fixed_size | sentence | semantic"),
    chunk_size: int = typer.Option(512, help="Max tokens per chunk (fixed_size strategy)"),
    recursive: bool = typer.Option(False, help="Recurse into sub-directories"),
) -> None:
    """Ingest document(s) into the vector store."""
    p = Path(path)
    if p.is_dir():
        docs = load_directory(p, recursive=recursive)
    else:
        docs = [load_document(p)]

    if not docs:
        console.print("[yellow]No supported documents found.[/yellow]")
        raise typer.Exit(1)

    store = _get_store()
    total_chunks = 0

    for doc in docs:
        console.print(f"\n[bold]Ingesting:[/bold] {doc.name} ({len(doc.text):,} chars)")
        chunks = chunk_document(
            doc.text,
            document_name=doc.name,
            strategy=strategy,
            chunk_size=chunk_size,
            metadata=doc.metadata,
        )
        embeddings = embed_texts([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)
        total_chunks += len(chunks)
        console.print(f"  → {len(chunks)} chunks stored (strategy: {strategy})")

    console.print(
        f"\n[green]✓ Ingestion complete:[/green] {len(docs)} document(s), "
        f"{total_chunks} total chunks stored in '{CHROMA_DB_PATH}'."
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language query"),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Number of chunks to retrieve"),
    min_score: float = typer.Option(MIN_RELEVANCE_SCORE, help="Minimum relevance score"),
    doc_filter: Optional[str] = typer.Option(None, help="Restrict search to this document name"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Query the system and get a structured decision-support response."""
    store = _get_store()
    if store.count() == 0:
        console.print("[red]Vector store is empty. Run 'ingest' first.[/red]")
        raise typer.Exit(1)

    retriever = ContextRetriever(
        vector_store=store,
        top_k=top_k,
        min_score=min_score,
    )

    with console.status("[bold green]Retrieving context…"):
        chunks_with_scores, usage = retriever.retrieve(question, document_filter=doc_filter)

    with console.status("[bold green]Generating response…"):
        response = generate_response(question, chunks_with_scores, usage)

    if json_output:
        console.print_json(response.model_dump_json(indent=2))
        return

    _render_response(response)


@app.command()
def status() -> None:
    """Show the current state of the vector store."""
    store = _get_store()
    count = store.count()
    docs = store.list_documents()

    table = Table(title="Vector Store Status", box=box.ROUNDED)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")
    table.add_row("Storage path", CHROMA_DB_PATH)
    table.add_row("Collection", COLLECTION_NAME)
    table.add_row("Total chunks", str(count))
    table.add_row("Documents", str(len(docs)))
    console.print(table)

    if docs:
        doc_table = Table(title="Ingested Documents", box=box.SIMPLE)
        doc_table.add_column("#", style="dim")
        doc_table.add_column("Document Name")
        for i, name in enumerate(docs, 1):
            doc_table.add_row(str(i), name)
        console.print(doc_table)


@app.command()
def clear() -> None:
    """Clear all data from the vector store."""
    typer.confirm("This will delete all stored chunks. Continue?", abort=True)
    store = _get_store()
    store.clear()
    console.print("[green]✓ Vector store cleared.[/green]")


@app.command()
def demo() -> None:
    """
    Run a full end-to-end demo using the bundled sample documents.

    This demonstrates the complete RAG pipeline without any configuration.
    """
    sample_dir = Path(__file__).parent / "sample_docs"
    if not sample_dir.exists() or not any(sample_dir.iterdir()):
        console.print("[red]sample_docs/ directory is empty or missing.[/red]")
        raise typer.Exit(1)

    console.rule("[bold blue]AI Decision Support System – RAG Demo[/bold blue]")
    console.print(
        "\nThis demo showcases:\n"
        "  • Document ingestion & chunking\n"
        "  • Embedding generation & vector storage\n"
        "  • Semantic search & context retrieval\n"
        "  • Structured generation with citations\n"
        "  • Risk indicators, confidence scores & sourced reasoning\n"
    )

    # Ingest
    console.rule("Step 1: Ingesting sample documents")
    store = _get_store()
    store.clear()
    docs = load_directory(sample_dir)
    total_chunks = 0
    for doc in docs:
        chunks = chunk_document(doc.text, document_name=doc.name, strategy="sentence")
        embeddings = embed_texts([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)
        total_chunks += len(chunks)
        console.print(f"  ✓ {doc.name}: {len(chunks)} chunks")
    console.print(f"\n[green]Total:[/green] {total_chunks} chunks from {len(docs)} documents\n")

    # Query
    demo_queries = [
        "What are the main financial risks identified in the reports?",
        "What mitigation strategies are recommended?",
        "What is the overall confidence in the investment decision?",
    ]

    retriever = ContextRetriever(vector_store=store)
    for i, q in enumerate(demo_queries, 1):
        console.rule(f"Step 2.{i}: Query")
        console.print(f"[bold cyan]Q:[/bold cyan] {q}\n")
        chunks_with_scores, usage = retriever.retrieve(q)
        response = generate_response(q, chunks_with_scores, usage)
        _render_response(response)
        console.print()


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------

def _risk_colour(level: RiskLevel) -> str:
    return {
        RiskLevel.LOW: "green",
        RiskLevel.MEDIUM: "yellow",
        RiskLevel.HIGH: "red",
        RiskLevel.CRITICAL: "bold red",
    }.get(level, "white")


def _render_response(response) -> None:
    """Pretty-print a DecisionResponse to the terminal."""

    # Header
    conf_colour = "green" if response.confidence_score >= 0.75 else (
        "yellow" if response.confidence_score >= 0.50 else "red"
    )
    console.print(
        Panel(
            f"[bold]{response.summary}[/bold]",
            title=f"[bold blue]Decision Support Analysis[/bold blue]  "
                  f"Confidence: [{conf_colour}]{response.confidence_score:.0%} "
                  f"({response.confidence_label()})[/{conf_colour}]",
            border_style="blue",
        )
    )

    # Answer
    console.print("\n[bold]Answer[/bold]")
    console.print(response.answer)

    # Reasoning
    if response.reasoning:
        console.print("\n[bold]Reasoning Chain[/bold]")
        for line in response.reasoning.strip().splitlines():
            console.print(f"  {line}")

    # Risk indicators
    if response.risk_indicators:
        console.print("\n[bold]Risk Indicators[/bold]")
        risk_table = Table(box=box.SIMPLE, show_header=True)
        risk_table.add_column("Category")
        risk_table.add_column("Level")
        risk_table.add_column("Score")
        risk_table.add_column("Description")
        for r in response.risk_indicators:
            colour = _risk_colour(r.level)
            risk_table.add_row(
                r.category,
                Text(r.level.upper(), style=colour),
                f"{r.score:.2f}",
                r.description[:80],
            )
        console.print(risk_table)

    # Citations
    if response.citations:
        console.print("\n[bold]Sources[/bold]")
        for i, c in enumerate(response.citations, 1):
            console.print(
                f"  [{i}] [cyan]{c.document_name}[/cyan] "
                f"(relevance: {c.relevance_score:.3f})\n"
                f"      \"{c.excerpt[:120]}…\""
            )

    # Hallucination flags
    if response.hallucination_flags:
        console.print("\n[bold yellow]⚠ Hallucination Flags[/bold yellow]")
        for flag in response.hallucination_flags:
            console.print(f"  • {flag}")

    # Context window usage
    if response.context_window_usage:
        u = response.context_window_usage
        console.print(
            f"\n[dim]Context window: {u.context_tokens_used} context tokens, "
            f"{u.utilization_pct:.1f}% utilization, "
            f"{u.chunks_included} chunks included"
            + (f", {u.chunks_truncated} truncated" if u.chunks_truncated else "")
            + "[/dim]"
        )


if __name__ == "__main__":
    app()
