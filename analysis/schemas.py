"""
Pydantic models and Anthropic tool definition for job fit analysis.
"""
from typing import Literal

from pydantic import BaseModel, Field


class RequirementMapping(BaseModel):
    requirement: str = Field(
        description="Key requirement from the JD, verbatim or closely paraphrased"
    )
    experience_match: str = Field(
        description=(
            "Specific Jeff experience that addresses this requirement. "
            "Name the project and company."
        )
    )
    match_strength: Literal["strong", "partial", "weak"] = Field(
        description=(
            "How well the experience matches: "
            "strong=direct parallel, partial=transferable, weak=adjacent"
        )
    )


class GutCheckRow(BaseModel):
    dimension: str = Field(
        description="Evaluation dimension name (must match a dimension from the rubric)"
    )
    assessment: Literal["strong_fit", "good_fit", "partial_fit", "gap"] = Field(
        description="Assessment for this dimension"
    )
    evidence: str = Field(
        description="One-line rationale citing a specific project or experience"
    )
    concern: str = Field(
        default="",
        description="Specific concern or gap if applicable, empty string if none",
    )


class JobFitAnalysis(BaseModel):
    fit_score: int = Field(
        ge=1, le=10,
        description="Overall fit score 1-10 per the rubric"
    )
    fit_rationale: str = Field(
        description=(
            "2-3 sentence direct explanation of the score. "
            "Be specific — name projects and companies. Don't be vague."
        )
    )
    top_requirements: list[RequirementMapping] = Field(
        min_length=3, max_length=5,
        description="3-5 key JD requirements mapped to Jeff's experience"
    )
    gut_check_table: list[GutCheckRow] = Field(
        description=(
            "Evaluation across each rubric dimension. "
            "Include one row per dimension defined in the rubric."
        )
    )
    lead_with: list[str] = Field(
        min_length=2, max_length=4,
        description=(
            "2-4 specific experiences or projects Jeff should lead with "
            "in the application and interviews. Be specific (project name + why)."
        )
    )
    watch_out_for: list[str] = Field(
        min_length=0, max_length=3,
        description="0-3 specific gaps or concerns Jeff should address proactively"
    )
    application_recommendation: Literal[
        "strongly_apply", "apply", "apply_with_caveats", "skip"
    ] = Field(description="Final recommendation on whether to pursue this role")


# Anthropic tool definition (mirrors JobFitAnalysis schema)
ANALYZE_JOB_FIT_TOOL: dict = {
    "name": "analyze_job_fit",
    "description": (
        "Return structured job fit analysis for a given job description "
        "and candidate experience. You MUST call this tool — do not output plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fit_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Overall fit score 1-10",
            },
            "fit_rationale": {
                "type": "string",
                "description": "2-3 sentence direct explanation. Name specific projects.",
            },
            "top_requirements": {
                "type": "array",
                "minItems": 3,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement": {"type": "string"},
                        "experience_match": {"type": "string"},
                        "match_strength": {
                            "type": "string",
                            "enum": ["strong", "partial", "weak"],
                        },
                    },
                    "required": ["requirement", "experience_match", "match_strength"],
                },
            },
            "gut_check_table": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dimension": {"type": "string"},
                        "assessment": {
                            "type": "string",
                            "enum": ["strong_fit", "good_fit", "partial_fit", "gap"],
                        },
                        "evidence": {"type": "string"},
                        "concern": {"type": "string"},
                    },
                    "required": ["dimension", "assessment", "evidence", "concern"],
                },
            },
            "lead_with": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {"type": "string"},
            },
            "watch_out_for": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string"},
            },
            "application_recommendation": {
                "type": "string",
                "enum": ["strongly_apply", "apply", "apply_with_caveats", "skip"],
            },
        },
        "required": [
            "fit_score",
            "fit_rationale",
            "top_requirements",
            "gut_check_table",
            "lead_with",
            "watch_out_for",
            "application_recommendation",
        ],
    },
}
