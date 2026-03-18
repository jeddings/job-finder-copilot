"""
Document loaders for .txt, .docx, and .pdf files.
"""
from pathlib import Path

from docx import Document
from pypdf import PdfReader

CONTENT_DIR = Path("/Users/jeddings/dev/job-finder/content")

# Source documents to ingest, keyed by source_type
SOURCE_FILES: dict[str, str] = {
    "work_narrative": "Jeff-Eddings-Work-and-Projects.txt",
    "resume": "Jeff Eddings - Resume - Canonical.pdf",
    "career_brief": "Jeff Eddings Career Operating Brief.docx",
    "skills": "Jeff Eddings Core Assets.docx",
    "positioning": "Jeff Eddings Positioning Variants.docx",
    "peer_feedback": "Jeff Eddings Peer Feedback.docx",
}


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_docx(path: Path) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_pdf(path: Path) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def load_source(source_type: str) -> tuple[str, Path]:
    """Load a source document by its source_type key. Returns (text, path)."""
    filename = SOURCE_FILES[source_type]
    path = CONTENT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_txt(path), path
    elif suffix == ".docx":
        return load_docx(path), path
    elif suffix == ".pdf":
        return load_pdf(path), path
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def load_all_sources() -> dict[str, str]:
    """Load all source documents. Returns {source_type: text}."""
    results = {}
    for source_type in SOURCE_FILES:
        text, path = load_source(source_type)
        results[source_type] = text
    return results
