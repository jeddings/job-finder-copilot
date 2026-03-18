# Job Search Copilot

AI-assisted job-fit analysis.

This is a personal learning project. I built it to learn practical AI engineering patterns in a real workflow: retrieval-augmented generation (RAG), prompt design, structured outputs, and simple product UX.

I am using GitHub Copilot as part of the learning process while I review and adapt the code.

The project evaluates how well a job description matches someone's background by combining:
- RAG over personal career documents
- A weighted scoring rubric
- Structured LLM output validated with Pydantic
- Streamlit UI + CLI workflows

## What It Does

Given a job description, the app:
1. Cleans and segments the JD (core responsibilities vs. qualifications boilerplate)
2. Retrieves the most relevant career evidence from a local Chroma vector store
3. Sends JD + retrieved evidence + rubric instructions to Claude
4. Forces structured JSON output through a tool schema
5. Validates and renders:
   - Fit score (1-10)
   - Fit rationale
   - Requirement-to-experience mapping
   - Gut-check table by rubric dimension
   - Lead-with points and watch-outs
   - Apply/skip recommendation

It also supports comparing 2-3 roles side by side.

## Why This Exists

I wanted a project that was useful in my current daily life and technically deep enough to teach me modern LLM app patterns.

This repo is intentionally iterative. I use it to test ideas, keep what works, and refactor as I learn.

## Current Capabilities

- Streamlit app for interactive analysis
- CLI for terminal-first workflows
- Parallel analysis for JD comparison
- Configurable rubric in YAML
- Local ChromaDB persistence
- Story extraction utility for STAR-style examples
- Retry / backoff handling for transient Anthropic overload errors (including HTTP 529)

## Project Intent

- Primary goal: learning by building and iterating
- Secondary goal: useful personal workflow for job search decisions
- Public visibility: shared as a working notebook/codebase, not as a polished product

## Tech Stack

- Python
- Streamlit
- Anthropic API
- ChromaDB
- LangChain ecosystem packages
- Pydantic
- Rich

## What I Am Practicing Here

- Building reliable LLM pipelines instead of one-shot prompts
- Using retrieval to ground outputs in concrete evidence
- Constraining output with schemas/tool calls and validating results
- Designing a simple UI/CLI pair around the same core engine
- Handling real-world provider behavior (retries, overloads, failures)

## Project Structure

```text
.
├── app.py                    # Streamlit UI
├── cli.py                    # CLI entrypoint
├── requirements.txt
├── config/
│   └── rubric.yaml           # Weighted scoring rubric
├── analysis/
│   ├── analyzer.py           # Main pipeline: clean -> retrieve -> analyze -> validate
│   ├── prompts.py            # System/user prompt builders
│   └── schemas.py            # Pydantic models + tool schema
├── rag/
│   └── retriever.py          # Chroma retrieval logic
├── ingest/
│   ├── loader.py             # Loads source docs
│   ├── chunker.py            # Chunking strategies by source type
│   └── embedder.py           # Chroma ingest + embeddings
├── scripts/
│   ├── run_ingest.py         # Build/reset vector store
│   └── extract_stories.py    # Extract STAR-like story records
└── utils/
    └── jd_cleaner.py         # JD cleanup + retrieval query extraction
```

## Prerequisites

- Python 3.10+
- Anthropic API key
- Career source documents available locally

## Setup

1. Clone and enter the repo

```bash
git clone https://github.com/jeddings/job-finder-copilot.git
cd job-finder-copilot
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables

```bash
cp .env.example .env
```

Set this value in `.env`:

```env
ANTHROPIC_API_KEY=your_key_here
```

## Configure Source Documents

This project expects source files listed in `ingest/loader.py`:
- Work and projects narrative
- Resume PDF
- Career operating brief
- Core skills/assets
- Positioning (resume) variants

### Important for Other Users

The loader currently references an absolute local path for content:

`/Users/jeddings/dev/job-finder/content`

If you run this in a different environment, update `CONTENT_DIR` in `ingest/loader.py` (or adapt it to use an environment variable).

This hardcoded path is acceptable for my own use right now, but it is one of the first things I would generalize if I wanted broader portability.

## Ingest Documents

Run an initial ingest before analysis:

```bash
python scripts/run_ingest.py --reset
```

This will:
- Load configured source docs
- Chunk them by source type
- Embed and store them in local ChromaDB (`data/chroma_db`)

## Run the App

### Streamlit UI (Primary)

```bash
streamlit run app.py
```

Then paste one JD into Analyze Role, or compare 2-3 roles in Compare Roles.

### CLI

Single JD from file:

```bash
python cli.py --jd-file path/to/jd.txt
```

Paste-mode single JD:

```bash
python cli.py
```

Compare multiple JDs:

```bash
python cli.py --compare jd1.txt jd2.txt jd3.txt
```

Use a different model/chunk count:

```bash
python cli.py --jd-file jd.txt --model claude-haiku-4-5-20251001 --n-chunks 8
```

## How Scoring Works

Scoring and calibration live in `config/rubric.yaml`.

Key ideas:
- Weighted dimensions (example: core role match, seniority, domain depth, differentiators)
- Explicit strengths/weaknesses calibration
- Fixed 1-10 score interpretation bands

You can adjust weights and criteria in the YAML. No re-ingest is needed for rubric-only changes.

## Retry Behavior and Overload Handling

The analyzer includes retry + exponential backoff for transient provider errors (for example Anthropic HTTP 529 overload).

In Streamlit Analyze Role, the UI shows a user-facing retry message while retries are in progress.

## Story Extraction Utility

Extract STAR-like records from work narrative chunks:

```bash
python scripts/extract_stories.py --dry-run
python scripts/extract_stories.py
```

- Dry run previews extraction quality/confidence
- Non-dry run writes to `jeff_story_bank` collection in Chroma

## Troubleshooting

### "ANTHROPIC_API_KEY not set"
- Ensure `.env` exists and has a valid key
- Restart terminal/session after changes

### "Overloaded" / HTTP 529 from Anthropic
- Retry is automatic
- If persistent, wait and rerun or switch to a smaller/faster model

### Import resolution warnings in editor
- Confirm VS Code is using the repo virtual environment
- Reinstall requirements in that environment

### Ingest cannot find files
- Verify source filenames and content directory in `ingest/loader.py`

## Privacy Notes

- Career documents and vector DB are local by default
- Job descriptions and retrieved evidence are sent to Anthropic for analysis
- Do not include sensitive content you do not want transmitted to the model provider

## Roadmap Ideas

- Make content directory configurable via environment variable
- Add tests for chunking/retrieval/analyzer behavior
- Add sample/synthetic data mode for easier external demos
- Add optional export of analyses to markdown/JSON files

## Notes for External Readers

- This repository is maintained as a personal learning codebase.
- It may change quickly, caveat emptor.
- You are welcome to read and adapt ideas, but there is no formal support commitment.
- If something looks rough around the edges, that is usually because I am optimizing for learning speed over polish.
- If you have any suggestions, they are most appreciated!
