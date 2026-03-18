#!/usr/bin/env python3
"""
Job Search Copilot — CLI

Usage:
    # Single JD analysis (paste mode):
    python cli.py

    # Single JD from file:
    python cli.py --jd-file path/to/jd.txt

    # Compare multiple JDs:
    python cli.py --compare jd1.txt jd2.txt jd3.txt

    # Use haiku for fast/cheap runs:
    python cli.py --model claude-haiku-4-5 --jd-file jd.txt
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from analysis.analyzer import DEFAULT_MODEL, analyze_job, analyze_jobs_parallel
from analysis.schemas import JobFitAnalysis

console = Console()

SCORE_STYLES = {
    (9, 10): ("bold green", "✦"),
    (7, 8): ("green", "✓"),
    (5, 6): ("yellow", "~"),
    (3, 4): ("orange3", "!"),
    (1, 2): ("red", "✗"),
}

RECOMMENDATION_LABELS = {
    "strongly_apply": ("[bold green]STRONGLY APPLY[/bold green]", "green"),
    "apply": ("[green]APPLY[/green]", "green"),
    "apply_with_caveats": ("[yellow]APPLY WITH CAVEATS[/yellow]", "yellow"),
    "skip": ("[red]SKIP[/red]", "red"),
}

ASSESSMENT_STYLES = {
    "strong_fit": ("[green]Strong Fit[/green]", "green"),
    "good_fit": ("[green]Good Fit[/green]", "green"),
    "partial_fit": ("[yellow]Partial Fit[/yellow]", "yellow"),
    "gap": ("[red]Gap[/red]", "red"),
}


def score_style(score: int) -> tuple[str, str]:
    for (lo, hi), style in SCORE_STYLES.items():
        if lo <= score <= hi:
            return style
    return ("white", "?")


def render_single_analysis(analysis: JobFitAnalysis, title: str = "Job Fit Analysis") -> None:
    style, icon = score_style(analysis.fit_score)
    rec_label, rec_style = RECOMMENDATION_LABELS.get(
        analysis.application_recommendation, ("[white]UNKNOWN[/white]", "white")
    )

    # Header panel
    console.print(Panel(
        f"[{style}]{icon} Fit Score: {analysis.fit_score}/10[/{style}]\n\n"
        f"{analysis.fit_rationale}\n\n"
        f"Recommendation: {rec_label}",
        title=title,
        border_style=rec_style,
        expand=False,
    ))

    # Requirements mapping table
    req_table = Table(
        title="Requirements Mapping",
        show_lines=True,
        box=box.ROUNDED,
        expand=True,
    )
    req_table.add_column("JD Requirement", style="cyan", ratio=3)
    req_table.add_column("Jeff's Experience", ratio=4)
    req_table.add_column("Strength", ratio=1)
    for r in analysis.top_requirements:
        strength_style = {"strong": "green", "partial": "yellow", "weak": "red"}.get(
            r.match_strength, "white"
        )
        req_table.add_row(
            r.requirement,
            r.experience_match,
            f"[{strength_style}]{r.match_strength.title()}[/{strength_style}]",
        )
    console.print(req_table)

    # Gut check table
    gut_table = Table(
        title="Gut Check",
        show_lines=True,
        box=box.ROUNDED,
        expand=True,
    )
    gut_table.add_column("Dimension", style="bold", ratio=2)
    gut_table.add_column("Assessment", ratio=1)
    gut_table.add_column("Evidence", ratio=4)
    gut_table.add_column("Concern", style="italic", ratio=2)
    for row in analysis.gut_check_table:
        label, _ = ASSESSMENT_STYLES.get(row.assessment, (row.assessment, "white"))
        gut_table.add_row(
            row.dimension,
            label,
            row.evidence,
            row.concern or "—",
        )
    console.print(gut_table)

    # Lead with / Watch out
    lead_items = "\n".join(f"  • {item}" for item in analysis.lead_with)
    console.print(Panel(lead_items, title="Lead With These", border_style="green", expand=False))

    if analysis.watch_out_for:
        watch_items = "\n".join(f"  • {item}" for item in analysis.watch_out_for)
        console.print(Panel(watch_items, title="Watch Out For", border_style="red", expand=False))

    console.print()


def render_comparison(results: list[tuple[str, JobFitAnalysis]]) -> None:
    if not results:
        return

    # Summary comparison table
    comp_table = Table(
        title="Role Comparison",
        show_lines=True,
        box=box.ROUNDED,
        expand=True,
    )
    comp_table.add_column("Dimension", style="bold", ratio=2)
    for label, _ in results:
        comp_table.add_column(label, ratio=2)

    # Fit score row
    score_cells = []
    for _, analysis in results:
        style, icon = score_style(analysis.fit_score)
        score_cells.append(f"[{style}]{icon} {analysis.fit_score}/10[/{style}]")
    comp_table.add_row("Fit Score", *score_cells)

    # Recommendation row
    rec_cells = []
    for _, analysis in results:
        rec_label, _ = RECOMMENDATION_LABELS.get(
            analysis.application_recommendation, (analysis.application_recommendation, "white")
        )
        rec_cells.append(rec_label)
    comp_table.add_row("Recommendation", *rec_cells)

    # Gut check dimensions (use first analysis as reference for dimension names)
    all_dimensions = [row.dimension for row in results[0][1].gut_check_table]
    for dim_name in all_dimensions:
        dim_cells = []
        for _, analysis in results:
            matching = next(
                (r for r in analysis.gut_check_table if r.dimension == dim_name), None
            )
            if matching:
                label, _ = ASSESSMENT_STYLES.get(matching.assessment, (matching.assessment, "white"))
                dim_cells.append(label)
            else:
                dim_cells.append("—")
        comp_table.add_row(dim_name, *dim_cells)

    # Top concern row
    concern_cells = []
    for _, analysis in results:
        concerns = [r.concern for r in analysis.gut_check_table if r.concern]
        concern_cells.append(concerns[0] if concerns else "—")
    comp_table.add_row("Top Concern", *concern_cells)

    console.print(comp_table)

    # Priority ranking
    sorted_results = sorted(results, key=lambda x: x[1].fit_score, reverse=True)
    ranking = " > ".join(f"{label} ({analysis.fit_score})" for label, analysis in sorted_results)
    console.print(Panel(
        f"[bold]Recommended priority:[/bold] {ranking}",
        border_style="blue",
        expand=False,
    ))
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="AI-assisted job fit analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--jd-file", type=Path, help="Path to a job description text file")
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        metavar="JD_FILE",
        help="Compare multiple JDs side-by-side",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=8,
        help="Number of experience chunks to retrieve (default: 8)",
    )
    args = parser.parse_args()

    # ── Compare mode ──────────────────────────────────────────────────────────
    if args.compare:
        jd_files = args.compare
        labels = [p.stem for p in jd_files]
        jd_texts = [p.read_text() for p in jd_files]

        console.print(f"\n[bold]Comparing {len(jd_files)} roles...[/bold]\n")
        with console.status(f"Analyzing {len(jd_files)} JDs in parallel..."):
            results = analyze_jobs_parallel(
                jd_texts, labels=labels, model=args.model, n_chunks=args.n_chunks
            )

        render_comparison(results)

        # Also show individual analyses
        for label, analysis in results:
            console.rule(f"[bold]{label}[/bold]")
            render_single_analysis(analysis, title=label)

        return

    # ── Single JD mode ────────────────────────────────────────────────────────
    if args.jd_file:
        jd_text = args.jd_file.read_text()
        title = args.jd_file.stem
    else:
        console.print("[dim]Paste job description below. Press Ctrl+D (or Ctrl+Z on Windows) when done.[/dim]\n")
        jd_text = sys.stdin.read()
        title = "Job Fit Analysis"

    if not jd_text.strip():
        console.print("[red]Error: empty job description.[/red]")
        sys.exit(1)

    with console.status(f"Analyzing with {args.model}..."):
        analysis = analyze_job(jd_text, model=args.model, n_chunks=args.n_chunks)

    render_single_analysis(analysis, title=title)


if __name__ == "__main__":
    main()
