#!/usr/bin/env python3
"""
Extract STAR story records from work_narrative chunks and optionally
write them to a separate ChromaDB collection (jeff_story_bank).

Usage:
    python scripts/extract_stories.py --dry-run   # preview extraction, no DB write
    python scripts/extract_stories.py              # extract and write to ChromaDB
"""
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from ingest.chunker import chunk_work_narrative
from ingest.loader import load_source

console = Console()

# Impact/result line markers, checked in order (first match wins).
# The document uses "* Impact: text" format (bullet + keyword), so we strip
# the leading bullet character before checking.
RESULT_MARKERS = [
    re.compile(r"^Impact:\s*", re.IGNORECASE),
    re.compile(r"^Results?:\s*", re.IGNORECASE),
    re.compile(r"^Outcomes?:\s*", re.IGNORECASE),
    re.compile(r"^Delivered:\s*", re.IGNORECASE),
    re.compile(r"^Achieved:\s*", re.IGNORECASE),
]

# Section separator line in the document
SECTION_SEPARATOR = re.compile(r"^\s*\*\s*\*\s*\*\s*$")

# Common themes to tag stories with (keyword-based inference)
THEME_KEYWORDS = {
    "developer_ecosystem": ["developer", "api", "sdk", "platform", "partner", "third-party", "ecosystem"],
    "reliability": ["reliability", "uptime", "sla", "incident", "oncall", "on-call", "availability"],
    "governance": ["governance", "compliance", "policy", "legal", "privacy", "trust", "safety"],
    "cross_org": ["cross-functional", "cross-org", "stakeholder", "alignment", "matrixed", "orchestrat"],
    "zero_to_one": ["0→1", "0 to 1", "launched", "new product", "greenfield", "from scratch", "built from"],
    "ml_ai": ["machine learning", "ml ", "ai ", "model", "ranking", "recommendation", "inference"],
    "data_platform": ["data", "pipeline", "warehouse", "analytics", "metrics", "dashboard"],
    "growth_scale": ["growth", "scale", "revenue", "monetization", "conversion", "engagement"],
}


@dataclass
class StoryRecord:
    title: str
    company: str
    action_summary: str        # First bullet / brief narrative
    result_text: str           # Extracted impact/result lines
    themes: list[str] = field(default_factory=list)
    confidence: str = "low"    # high | medium | low
    full_text: str = ""


def infer_themes(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        theme
        for theme, keywords in THEME_KEYWORDS.items()
        if any(kw in text_lower for kw in keywords)
    ]


def strip_bullet(line: str) -> str:
    """Strip leading bullet characters (*, -, •) from a line."""
    return re.sub(r"^\s*[*\-•]\s*", "", line).strip()


def extract_result(lines: list[str]) -> tuple[str, str]:
    """
    Find result/impact text from a list of lines.
    The document uses "* Impact: text" format, so we strip the bullet before matching.
    Returns (result_text, confidence).
    confidence: high=found labeled marker, medium=fallback marker, low=last bullet
    """
    impact_lines = []

    for i, marker in enumerate(RESULT_MARKERS):
        for line in lines:
            stripped = strip_bullet(line)
            if marker.match(stripped):
                result = marker.sub("", stripped).strip()
                confidence = "high" if i == 0 else "medium"
                impact_lines.append(result)

        if impact_lines:
            # Return all impact lines for this marker (there may be multiple per story)
            return " | ".join(impact_lines), ("high" if i == 0 else "medium")

    # Last resort: use the last non-empty bullet line (excluding separator lines)
    content_lines = [
        strip_bullet(l) for l in lines
        if l.strip() and not SECTION_SEPARATOR.match(l)
    ]
    if content_lines:
        return content_lines[-1], "low"

    return "", "low"


def extract_stories_from_chunks(chunks) -> list[StoryRecord]:
    stories = []
    for chunk in chunks:
        lines = chunk.text.split("\n")
        company = chunk.metadata.get("company", "")
        title = chunk.metadata.get("section", "")

        # Skip header lines, empty lines, and section separators ("* * *")
        content_lines = [
            l for l in lines
            if not l.startswith("Company/Org:")
            and not l.startswith("Project/Area:")
            and l.strip()
            and not SECTION_SEPARATOR.match(l)
        ]

        if not content_lines:
            continue

        # Action summary: first meaningful non-impact line
        first_line = strip_bullet(content_lines[0])
        # Skip if first line is an Impact: line
        if any(m.match(first_line) for m in RESULT_MARKERS):
            first_line = strip_bullet(content_lines[1]) if len(content_lines) > 1 else first_line
        action_summary = first_line

        # Extract result
        result_text, confidence = extract_result(content_lines)

        # Infer themes
        themes = infer_themes(chunk.text)

        stories.append(StoryRecord(
            title=title,
            company=company,
            action_summary=action_summary[:120] + ("..." if len(action_summary) > 120 else ""),
            result_text=result_text[:200] + ("..." if len(result_text) > 200 else ""),
            themes=themes,
            confidence=confidence,
            full_text=chunk.text,
        ))

    return stories


def write_stories_to_chroma(stories: list[StoryRecord]) -> int:
    import chromadb
    from chromadb.utils import embedding_functions
    from ingest.embedder import CHROMA_PATH, EMBEDDING_MODEL

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = embedding_functions.DefaultEmbeddingFunction()

    try:
        client.delete_collection("jeff_story_bank")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="jeff_story_bank",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        documents=[s.full_text for s in stories],
        metadatas=[
            {
                "title": s.title,
                "company": s.company,
                "themes": ",".join(s.themes),
                "confidence": s.confidence,
                "result_text": s.result_text,
            }
            for s in stories
        ],
        ids=[f"story_{i:03d}" for i in range(len(stories))],
    )

    return len(stories)


def main():
    parser = argparse.ArgumentParser(description="Extract STAR stories from career narrative")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview extracted stories without writing to ChromaDB",
    )
    args = parser.parse_args()

    console.print("\n[bold]Job Search Copilot — Story Extraction[/bold]\n")

    # Load and chunk work narrative
    with console.status("Loading work narrative..."):
        text, _ = load_source("work_narrative")
        chunks = chunk_work_narrative(text)

    console.print(f"  [green]✓[/green] {len(chunks)} narrative chunks loaded\n")

    # Extract stories
    with console.status("Extracting STAR records..."):
        stories = extract_stories_from_chunks(chunks)

    console.print(f"  [green]✓[/green] {len(stories)} stories extracted\n")

    # Confidence summary
    from collections import Counter
    conf_counts = Counter(s.confidence for s in stories)
    console.print(
        f"  Confidence: [green]{conf_counts['high']} high[/green], "
        f"[yellow]{conf_counts['medium']} medium[/yellow], "
        f"[red]{conf_counts['low']} low[/red]\n"
    )

    # Display table
    table = Table(
        title=f"Extracted Stories ({'DRY RUN — no DB write' if args.dry_run else 'Preview'})",
        show_lines=True,
    )
    table.add_column("#", width=3)
    table.add_column("Title", style="cyan", width=28)
    table.add_column("Company", width=20)
    table.add_column("Action Summary", width=35)
    table.add_column("Result", width=35)
    table.add_column("Themes", width=20)
    table.add_column("Conf", width=6)

    for i, s in enumerate(stories, 1):
        conf_style = {"high": "green", "medium": "yellow", "low": "red"}.get(s.confidence, "white")
        table.add_row(
            str(i),
            s.title,
            s.company,
            s.action_summary,
            s.result_text,
            ", ".join(s.themes[:3]) or "—",
            f"[{conf_style}]{s.confidence}[/{conf_style}]",
        )

    console.print(table)

    if args.dry_run:
        console.print(
            "\n[yellow]Dry run complete. No changes written to ChromaDB.[/yellow]\n"
            "Review the table above, then run without --dry-run to commit.\n"
        )
        # Flag low-confidence extractions for manual review
        low_conf = [s for s in stories if s.confidence == "low"]
        if low_conf:
            console.print(
                f"[red]⚠ {len(low_conf)} low-confidence extractions may need manual review:[/red]"
            )
            for s in low_conf:
                console.print(f"  • {s.company} / {s.title}")
    else:
        with console.status("Writing to ChromaDB (jeff_story_bank)..."):
            n = write_stories_to_chroma(stories)
        console.print(f"\n[green]✓[/green] {n} stories written to jeff_story_bank collection.\n")


if __name__ == "__main__":
    main()
