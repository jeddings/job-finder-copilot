"""
Job Search Copilot — Streamlit UI (primary interface)

Run with:
    cd /Users/jeddings/dev/job-finder/copilot
    streamlit run app.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is in path when running via streamlit
sys.path.insert(0, str(Path(__file__).parent))

from analysis.analyzer import DEFAULT_MODEL, analyze_job, analyze_jobs_parallel
from analysis.schemas import JobFitAnalysis

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Job Search Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

SCORE_COLORS = {
    (9, 10): "#22c55e",
    (7, 8): "#86efac",
    (5, 6): "#fbbf24",
    (3, 4): "#f97316",
    (1, 2): "#ef4444",
}

ASSESSMENT_COLORS = {
    "strong_fit": "🟢",
    "good_fit": "🟢",
    "partial_fit": "🟡",
    "gap": "🔴",
}

REC_DISPLAY = {
    "strongly_apply": "✦ Strongly Apply",
    "apply": "✓ Apply",
    "apply_with_caveats": "~ Apply with Caveats",
    "skip": "✗ Skip",
}

REC_COLORS = {
    "strongly_apply": "success",
    "apply": "success",
    "apply_with_caveats": "warning",
    "skip": "error",
}


def score_color(score: int) -> str:
    for (lo, hi), color in SCORE_COLORS.items():
        if lo <= score <= hi:
            return color
    return "#ffffff"


def render_analysis(analysis: JobFitAnalysis, label: str = "") -> None:
    """Render a single job fit analysis."""
    col1, col2 = st.columns([1, 4])

    with col1:
        color = score_color(analysis.fit_score)
        st.markdown(
            f"""<div style="background:{color};border-radius:12px;padding:20px;text-align:center;">
            <div style="font-size:2.5rem;font-weight:bold;color:#111;">{analysis.fit_score}</div>
            <div style="font-size:0.8rem;color:#333;">out of 10</div>
            </div>""",
            unsafe_allow_html=True,
        )
        rec = analysis.application_recommendation
        rec_type = REC_COLORS.get(rec, "info")
        getattr(st, rec_type)(REC_DISPLAY.get(rec, rec))

    with col2:
        st.markdown(analysis.fit_rationale)

    st.markdown("---")

    # Two-column layout for tables
    col_req, col_gut = st.columns(2)

    with col_req:
        st.subheader("Requirements Mapping")
        for r in analysis.top_requirements:
            badge = {"strong": "🟢", "partial": "🟡", "weak": "🔴"}.get(r.match_strength, "")
            with st.expander(f"{badge} {r.requirement}", expanded=True):
                st.write(r.experience_match)
                st.caption(f"Match: **{r.match_strength}**")

    with col_gut:
        st.subheader("Gut Check")
        gut_data = [
            {
                "Dimension": row.dimension,
                "Assessment": f"{ASSESSMENT_COLORS.get(row.assessment, '')} {row.assessment.replace('_', ' ').title()}",
                "Evidence": row.evidence,
                "Concern": row.concern or "—",
            }
            for row in analysis.gut_check_table
        ]
        st.dataframe(
            pd.DataFrame(gut_data),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # Lead with / Watch out
    col_lead, col_watch = st.columns(2)

    with col_lead:
        st.subheader("Lead With These")
        for item in analysis.lead_with:
            st.success(item)

    with col_watch:
        st.subheader("Watch Out For")
        if analysis.watch_out_for:
            for item in analysis.watch_out_for:
                st.warning(item)
        else:
            st.info("No major concerns identified.")


def render_comparison(results: list[tuple[str, JobFitAnalysis]]) -> None:
    """Render side-by-side comparison of multiple JDs."""
    st.subheader("Side-by-Side Comparison")

    # Build comparison dataframe
    rows = []
    dimensions = [r.dimension for r in results[0][1].gut_check_table]

    for dim in dimensions:
        row = {"Dimension": dim}
        for label, analysis in results:
            matching = next(
                (r for r in analysis.gut_check_table if r.dimension == dim), None
            )
            if matching:
                icon = ASSESSMENT_COLORS.get(matching.assessment, "")
                row[label] = f"{icon} {matching.assessment.replace('_', ' ').title()}"
            else:
                row[label] = "—"
        rows.append(row)

    # Add fit score row at top
    score_row = {"Dimension": "Fit Score"}
    for label, analysis in results:
        score_row[label] = f"{analysis.fit_score}/10"
    rows.insert(0, score_row)

    # Add recommendation row
    rec_row = {"Dimension": "Recommendation"}
    for label, analysis in results:
        rec_row[label] = REC_DISPLAY.get(analysis.application_recommendation, "—")
    rows.insert(1, rec_row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Priority ranking
    sorted_results = sorted(results, key=lambda x: x[1].fit_score, reverse=True)
    ranking_text = " → ".join(
        f"**{label}** ({analysis.fit_score})" for label, analysis in sorted_results
    )
    st.info(f"Recommended priority: {ranking_text}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox(
        "Claude model",
        options=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        index=0,
        help="Sonnet = best quality. Haiku = faster & cheaper for quick checks.",
    )

    n_chunks = st.slider(
        "Experience chunks to retrieve",
        min_value=4,
        max_value=16,
        value=8,
        step=2,
        help="More chunks = more context for Claude, but slower and more expensive.",
    )

    show_raw = st.checkbox("Show raw JSON output", value=False)

    st.divider()

    st.markdown("**Documents**")
    if st.button("Re-ingest documents", help="Clear and rebuild the vector store from current files"):
        with st.spinner("Re-ingesting..."):
            result = subprocess.run(
                [sys.executable, "scripts/run_ingest.py", "--reset"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
        if result.returncode == 0:
            st.success("Re-ingested successfully.")
        else:
            st.error(f"Ingest failed:\n{result.stderr}")

    st.divider()
    st.caption("Career documents: `/dev/job-finder/content/`")
    st.caption("Rubric: `config/rubric.yaml`")

# ── Main tabs ─────────────────────────────────────────────────────────────────

st.title("🎯 Job Search Copilot")
st.caption("AI-powered fit analysis grounded in your career documents")

tab_single, tab_compare, tab_about = st.tabs(["Analyze Role", "Compare Roles", "About"])

# ── Tab 1: Single role analysis ────────────────────────────────────────────────

with tab_single:
    jd_text = st.text_area(
        "Paste Job Description",
        height=280,
        placeholder="Paste the full job description here — responsibilities, requirements, preferred qualifications...",
        key="jd_single",
    )

    if st.button("Analyze Fit ▶", type="primary", key="analyze_single") and jd_text.strip():
        with st.spinner(f"Retrieving experience and analyzing with {model}..."):
            retry_notice = st.empty()
            retry_state = {"count": 0}

            def _on_retry(attempt: int, max_attempts: int, sleep_seconds: float, _exc: Exception) -> None:
                retry_state["count"] = attempt
                retry_notice.warning(
                    f"Claude is temporarily overloaded. Retrying ({attempt}/{max_attempts - 1}) in {sleep_seconds:.1f}s..."
                )

            try:
                analysis = analyze_job(
                    jd_text,
                    model=model,
                    n_chunks=n_chunks,
                    on_retry=_on_retry,
                )
                st.session_state["last_analysis"] = analysis
                if retry_state["count"] > 0:
                    retry_notice.info(
                        f"Recovered after {retry_state['count']} retry attempt(s)."
                    )
                else:
                    retry_notice.empty()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

    if "last_analysis" in st.session_state:
        render_analysis(st.session_state["last_analysis"])

        if show_raw:
            with st.expander("Raw JSON Output"):
                st.json(st.session_state["last_analysis"].model_dump())

# ── Tab 2: Compare roles ────────────────────────────────────────────────────

with tab_compare:
    st.markdown("Paste 2-3 job descriptions to get a side-by-side fit comparison.")

    n_roles = st.radio("Number of roles to compare", [2, 3], horizontal=True, index=0)

    jd_inputs = []
    cols = st.columns(n_roles)
    for i, col in enumerate(cols):
        with col:
            label = st.text_input(f"Role {i+1} label", value=f"Role {i+1}", key=f"label_{i}")
            jd = st.text_area(
                f"Job Description {i+1}",
                height=220,
                placeholder=f"Paste JD {i+1} here...",
                key=f"jd_{i}",
            )
            jd_inputs.append((label, jd))

    if st.button("Compare Roles ▶", type="primary", key="analyze_compare"):
        valid = [(label, jd) for label, jd in jd_inputs if jd.strip()]
        if len(valid) < 2:
            st.warning("Please paste at least 2 job descriptions.")
        else:
            labels = [label for label, _ in valid]
            jd_texts = [jd for _, jd in valid]

            with st.spinner(f"Analyzing {len(valid)} roles in parallel..."):
                try:
                    results = analyze_jobs_parallel(
                        jd_texts, labels=labels, model=model, n_chunks=n_chunks
                    )
                    st.session_state["last_comparison"] = results
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
                    st.stop()

    if "last_comparison" in st.session_state:
        results = st.session_state["last_comparison"]
        render_comparison(results)

        st.markdown("---")
        st.subheader("Individual Analyses")
        for label, analysis in results:
            with st.expander(f"{label} — {analysis.fit_score}/10 · {REC_DISPLAY.get(analysis.application_recommendation, '')}"):
                render_analysis(analysis, label=label)

# ── Tab 3: About ─────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
## How it works

1. **Ingest**: Your career documents (Career Operating Brief, Core Assets, Work & Projects, Positioning, Peer Feedback, Resume) are chunked and embedded into a local ChromaDB vector store using `all-MiniLM-L6-v2` sentence embeddings.

2. **Retrieve**: When you paste a job description, the core responsibilities section is used to semantically retrieve the most relevant chunks of your career experience.

3. **Analyze**: Claude is called with a structured rubric prompt and forced tool use — it must return structured JSON, never prose. Pydantic validates the output.

4. **Output**: Fit score (1-10), requirements mapping, gut-check table, lead-with recommendations, and an application recommendation.

## Updating your documents

When you update a source document (Career Operating Brief, Positioning Variants, etc.):
1. Replace the file in `/Users/jeddings/dev/job-finder/content/`
2. Click **Re-ingest documents** in the sidebar

When you update rubric weights:
1. Edit `config/rubric.yaml`
2. No re-ingest needed — the rubric reloads on every analysis

## Commands

```bash
# Ingest documents
python scripts/run_ingest.py --reset

# Story extraction (preview first)
python scripts/extract_stories.py --dry-run
python scripts/extract_stories.py

# CLI analysis
python cli.py --jd-file my_jd.txt
python cli.py --compare jd1.txt jd2.txt jd3.txt
```
""")
