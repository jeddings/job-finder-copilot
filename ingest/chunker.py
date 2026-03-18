"""
Chunking strategies for career documents.

- Work-and-Projects.txt: structural chunking on ### headings
- DOCX/PDF: sliding paragraph-window chunking
"""
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_work_narrative(text: str, source_file: str = "Jeff-Eddings-Work-and-Projects.txt") -> list[Chunk]:
    """
    Structural chunking on ## (company) and ### (project) headings.
    Each ### section becomes one chunk, prefixed with its parent company context.
    """
    chunks: list[Chunk] = []
    current_company = "Unknown"
    current_section: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_section and current_lines:
            body = "\n".join(line for line in current_lines if line.strip())
            if not body.strip():
                return
            text_block = (
                f"Company/Org: {current_company}\n"
                f"Project/Area: {current_section}\n\n"
                f"{body}"
            )
            chunks.append(Chunk(
                text=text_block,
                metadata={
                    "source_file": source_file,
                    "source_type": "work_narrative",
                    "company": current_company,
                    "section": current_section,
                },
            ))

    for line in text.split("\n"):
        if line.startswith("## "):
            flush()
            current_company = line[3:].strip()
            current_section = None
            current_lines = []
        elif line.startswith("### "):
            flush()
            current_section = line[4:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return chunks


def chunk_by_paragraphs(
    text: str,
    source_type: str,
    source_file: str,
    max_words: int = 500,
    overlap_paragraphs: int = 1,
) -> list[Chunk]:
    """
    Sliding paragraph-window chunking for DOCX and PDF files.
    Groups paragraphs until max_words is reached, then carries
    the last `overlap_paragraphs` paragraphs into the next window.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    window: list[str] = []
    word_count = 0

    for para in paragraphs:
        words = len(para.split())
        if word_count + words > max_words and window:
            chunks.append(Chunk(
                text="\n\n".join(window),
                metadata={"source_type": source_type, "source_file": source_file},
            ))
            window = window[-overlap_paragraphs:] if overlap_paragraphs else []
            word_count = sum(len(p.split()) for p in window)

        window.append(para)
        word_count += words

    if window:
        chunks.append(Chunk(
            text="\n\n".join(window),
            metadata={"source_type": source_type, "source_file": source_file},
        ))

    return chunks


def chunk_resume_by_role(text: str, source_file: str = "Jeff Eddings - Resume - Canonical.pdf") -> list[Chunk]:
    """
    Splits resume PDF text into per-role chunks.
    Heuristic: new role starts when a line looks like a job title + company + date pattern,
    or when a year appears at the start of a line. Falls back to paragraph chunking if
    structural parsing yields fewer than 2 chunks.
    """
    import re

    # Try to split on year-like lines (e.g., "Meta  2020–2024" or "2020 - 2024")
    year_pattern = re.compile(r"^\s*(20\d{2}|19\d{2})\s*[–\-–—]", re.MULTILINE)
    # Also split on lines that are ALL CAPS (common for company names in resumes)
    caps_pattern = re.compile(r"^[A-Z][A-Z\s,\.&]+$", re.MULTILINE)

    lines = text.split("\n")
    role_chunks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Start a new role chunk when we see a year range or all-caps company name
        # after we've already accumulated some content
        is_boundary = (
            (year_pattern.match(stripped) or caps_pattern.match(stripped))
            and len(current) > 5
        )
        if is_boundary:
            if current:
                role_chunks.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        role_chunks.append(current)

    # Fall back to paragraph chunking if structural split didn't work well
    if len(role_chunks) < 2:
        return chunk_by_paragraphs(text, "resume", source_file, max_words=300)

    return [
        Chunk(
            text="\n".join(rc).strip(),
            metadata={"source_type": "resume", "source_file": source_file},
        )
        for rc in role_chunks
        if "\n".join(rc).strip()
    ]


def chunk_all_sources(sources: dict[str, str]) -> list[Chunk]:
    """
    Chunk all loaded sources using the appropriate strategy.
    sources: {source_type: text}
    Returns a flat list of all chunks.
    """
    all_chunks: list[Chunk] = []

    dispatch = {
        "work_narrative": lambda text: chunk_work_narrative(text),
        "resume": lambda text: chunk_resume_by_role(text),
        "career_brief": lambda text: chunk_by_paragraphs(text, "career_brief", "Jeff Eddings Career Operating Brief.docx"),
        "skills": lambda text: chunk_by_paragraphs(text, "skills", "Jeff Eddings Core Assets.docx"),
        "positioning": lambda text: chunk_by_paragraphs(text, "positioning", "Jeff Eddings Positioning Variants.docx"),
        "peer_feedback": lambda text: chunk_by_paragraphs(text, "peer_feedback", "Jeff Eddings Peer Feedback.docx", max_words=250),
    }

    for source_type, text in sources.items():
        fn = dispatch.get(source_type)
        if fn:
            chunks = fn(text)
            all_chunks.extend(chunks)

    return all_chunks
