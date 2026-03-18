"""
JD cleaning: split job description into core content vs. qualifications boilerplate.

The core section (responsibilities, mission, what you'll do) is used for
retrieval embedding. The qualifications are included in the analysis prompt
but don't dominate the embedding query.
"""
import re

# Section headers that typically start boilerplate / requirements sections
QUALIFICATIONS_HEADERS = re.compile(
    r"^\s*(minimum qualifications?|basic qualifications?|required qualifications?|"
    r"requirements?|what you.ll need|what we.re looking for|qualifications?|"
    r"preferred qualifications?|nice to have|bonus points?|"
    r"about the company|about us|equal opportunity|"
    r"compensation|benefits|why join us|our benefits)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Headers that mark the "preferred" / "nice-to-have" section
PREFERRED_HEADERS = re.compile(
    r"^\s*(preferred qualifications?|nice to have|bonus points?|"
    r"preferred skills?|plus if you have|would be a plus)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_jd(jd_text: str) -> dict[str, str]:
    """
    Split a job description into structured sections.

    Returns:
        {
            "core": str,          # responsibilities, mission, what you'll do
            "requirements": str,  # minimum/required qualifications
            "preferred": str,     # preferred/nice-to-have qualifications
            "full": str,          # full original text
        }

    The "core" field is what should be used for retrieval embedding.
    All fields are passed to the analysis prompt for completeness.
    """
    lines = jd_text.split("\n")
    sections = {"core": [], "requirements": [], "preferred": []}
    current_section = "core"

    for line in lines:
        stripped = line.strip()

        # Check for preferred section header first (subset of qualifications)
        if PREFERRED_HEADERS.match(stripped):
            current_section = "preferred"
            continue

        # Check for general qualifications / requirements section
        if QUALIFICATIONS_HEADERS.match(stripped):
            # Only switch to requirements if we're still in core
            # (don't switch back from preferred to requirements)
            if current_section == "core":
                current_section = "requirements"
            continue

        sections[current_section].append(line)

    return {
        "core": "\n".join(sections["core"]).strip(),
        "requirements": "\n".join(sections["requirements"]).strip(),
        "preferred": "\n".join(sections["preferred"]).strip(),
        "full": jd_text.strip(),
    }


def get_retrieval_query(jd_text: str) -> str:
    """
    Extract the best query string for retrieval from a JD.
    Uses the core section if available, falls back to full text.
    """
    cleaned = clean_jd(jd_text)
    core = cleaned["core"]
    # If the core section is too short (< 100 chars), the split probably failed;
    # fall back to the full text
    if len(core) >= 100:
        return core
    return cleaned["full"]
