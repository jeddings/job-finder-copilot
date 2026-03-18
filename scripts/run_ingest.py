#!/usr/bin/env python3
"""
Ingest career documents into ChromaDB.

Usage:
    python scripts/run_ingest.py           # ingest (skip if collection exists)
    python scripts/run_ingest.py --reset   # clear and re-ingest from scratch
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from ingest.chunker import chunk_all_sources
from ingest.embedder import collection_count, ingest_chunks
from ingest.loader import SOURCE_FILES, load_all_sources

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ingest career documents into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the existing collection and re-ingest from scratch",
    )
    args = parser.parse_args()

    console.print("\n[bold]Job Search Copilot — Document Ingest[/bold]\n")

    # Show source files
    table = Table(title="Source Documents", show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("File")
    for source_type, filename in SOURCE_FILES.items():
        path = Path("/Users/jeddings/dev/job-finder/content") / filename
        exists = "✓" if path.exists() else "[red]✗ MISSING[/red]"
        table.add_row(source_type, f"{filename}  {exists}")
    console.print(table)

    if args.reset:
        console.print("\n[yellow]--reset flag set: clearing existing collection...[/yellow]")

    # Load documents
    console.print("\nLoading documents...")
    with console.status("Reading files..."):
        sources = load_all_sources()

    for source_type, text in sources.items():
        word_count = len(text.split())
        console.print(f"  [green]✓[/green] {source_type}: {word_count:,} words")

    # Chunk
    console.print("\nChunking documents...")
    with console.status("Chunking..."):
        chunks = chunk_all_sources(sources)

    console.print(f"  [green]✓[/green] {len(chunks)} total chunks")

    # Show chunk distribution by source type
    from collections import Counter
    dist = Counter(c.metadata.get("source_type", "unknown") for c in chunks)
    for source_type, count in sorted(dist.items()):
        console.print(f"    {source_type}: {count}")

    # Ingest
    console.print(f"\n{'Re-ingesting' if args.reset else 'Ingesting'} into ChromaDB...")
    with console.status("Writing to vector store..."):
        n = ingest_chunks(chunks, reset=args.reset)

    total = collection_count()
    console.print(f"  [green]✓[/green] {n} chunks ingested. Collection total: {total}\n")


if __name__ == "__main__":
    main()
