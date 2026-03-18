"""
Prompt builders for job fit analysis.
System prompt is generated dynamically from config/rubric.yaml.
"""
from pathlib import Path

import yaml

RUBRIC_PATH = Path(__file__).parent.parent / "config" / "rubric.yaml"


def load_rubric(rubric_path: Path = RUBRIC_PATH) -> dict:
    with open(rubric_path) as f:
        return yaml.safe_load(f)


def build_system_prompt(rubric_path: Path = RUBRIC_PATH) -> str:
    rubric = load_rubric(rubric_path)

    # Build dimension descriptions with weights
    dim_lines = []
    for dim in rubric["dimensions"]:
        weight_pct = int(dim["weight"] * 100)
        dim_lines.append(f"  {len(dim_lines)+1}. **{dim['name']}** ({weight_pct}%): {dim['description']}")
    dimensions_block = "\n".join(dim_lines)

    # Build scoring scale
    scale_lines = []
    for band, desc in rubric["scoring_scale"].items():
        scale_lines.append(f"  - {band}: {desc}")
    scale_block = "\n".join(scale_lines)

    # Build calibration notes
    cal = rubric["calibration"]
    strengths = "\n".join(f"    - {s}" for s in cal["strengths"])
    weaknesses = "\n".join(f"    - {w}" for w in cal["weaknesses"])

    return f"""You are an expert career advisor and job fit analyst specializing in technical product management, technical program management, and platform leadership roles.

You have access to Jeff Eddings' career documents — work narratives, skills, peer feedback, positioning, and resume. Your job is to rigorously evaluate how well Jeff's experience matches a given job description and provide actionable, specific analysis.

## Evaluation Rubric

Weight these dimensions when computing the fit score:

{dimensions_block}

## Calibration Notes

Jeff's **core strengths** — weight these heavily when present in a JD:
{strengths}

Jeff's **relative weaknesses** — flag as concerns when a JD requires these:
{weaknesses}

Jeff's **superpower**: {cal['superpower']}

## Scoring Scale

{scale_block}

## Output Contract

You MUST use the `analyze_job_fit` tool to return your analysis. Do NOT output plain text or prose — only call the tool with structured data.

When citing evidence:
- Name specific projects and companies (e.g., "Instagram Developer Platform at Meta", not "a developer platform role")
- Reference actual accomplishments from the provided experience chunks
- Do not invent experience that isn't in the retrieved chunks
- Be direct and calibrated — avoid hedging language like "may be able to" or "could potentially"
"""


def build_user_prompt(
    jd_text: str,
    retrieved_chunks: list[dict],
    jd_sections: dict | None = None,
) -> str:
    """
    Build the user-turn prompt with the JD and retrieved experience chunks.

    jd_sections: optional output of jd_cleaner.clean_jd() for structured JD context
    """
    # Format JD section
    if jd_sections and (jd_sections.get("requirements") or jd_sections.get("preferred")):
        jd_block = f"""## Job Description — Core Responsibilities

{jd_sections['core']}
"""
        if jd_sections.get("requirements"):
            jd_block += f"""
## Minimum Qualifications

{jd_sections['requirements']}
"""
        if jd_sections.get("preferred"):
            jd_block += f"""
## Preferred Qualifications

{jd_sections['preferred']}
"""
    else:
        jd_block = f"""## Job Description

{jd_text}
"""

    # Format experience chunks
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        company = meta.get("company", "")
        section = meta.get("section", "")
        source_type = meta.get("source_type", "unknown")
        dist = chunk.get("distance", 0)
        label = f"Chunk {i} | {source_type}"
        if company:
            label += f" | {company}"
        if section:
            label += f" | {section}"
        label += f" | relevance={1-dist:.2f}"
        context_blocks.append(f"[{label}]\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_blocks)

    return f"""{jd_block}

## Retrieved Career Experience

The following chunks were retrieved from Jeff's career documents based on semantic similarity to the job description. Use these as the primary evidence base for your analysis.

{context_block}

## Instructions

Analyze the job description against Jeff's experience above. Call `analyze_job_fit` with your structured analysis. Cite specific project names and companies from the chunks. Be direct and calibrated in your scoring."""
