"""
Core analysis engine: RAG retrieval + Claude API call with forced tool use.
"""
import os
import random
import time
from pathlib import Path
from typing import Callable

import anthropic
from dotenv import load_dotenv

from analysis.prompts import RUBRIC_PATH, build_system_prompt, build_user_prompt
from analysis.schemas import ANALYZE_JOB_FIT_TOOL, JobFitAnalysis
from rag.retriever import retrieve_relevant_chunks
from utils.jd_cleaner import clean_jd, get_retrieval_query

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

DEFAULT_MODEL = "claude-sonnet-4-6"


def _is_transient_anthropic_error(exc: Exception) -> bool:
    """Return True for overload/rate-limit/network errors worth retrying."""
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504, 529}:
        return True

    # Anthropic SDK exceptions may carry structured response payloads.
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        err_type = err.get("type")
        if err_type in {"overloaded_error", "rate_limit_error", "api_error"}:
            return True

    # Network/connectivity exceptions are also transient.
    return isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError))


def _call_anthropic_with_retry(
    client: anthropic.Anthropic,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_attempts: int = 4,
    on_retry: Callable[[int, int, float, Exception], None] | None = None,
) -> anthropic.types.Message:
    """
    Call Anthropic with exponential backoff for transient provider errors.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                tools=[ANALYZE_JOB_FIT_TOOL],
                tool_choice={"type": "tool", "name": "analyze_job_fit"},
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            last_error = exc
            is_retryable = _is_transient_anthropic_error(exc)
            if not is_retryable or attempt == max_attempts:
                raise

            # Exponential backoff with jitter to avoid retry storms.
            sleep_seconds = min(1.2 * (2 ** (attempt - 1)), 10.0) + random.uniform(0, 0.4)
            if on_retry is not None:
                on_retry(attempt, max_attempts, sleep_seconds, exc)
            time.sleep(sleep_seconds)

    # Defensive fallback (loop should have returned or raised).
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected retry state in Anthropic call.")


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to copilot/.env or your environment."
        )
    return anthropic.Anthropic(api_key=api_key)


def analyze_job(
    jd_text: str,
    model: str = DEFAULT_MODEL,
    n_chunks: int = 8,
    rubric_path: Path = RUBRIC_PATH,
    on_retry: Callable[[int, int, float, Exception], None] | None = None,
) -> JobFitAnalysis:
    """
    Full pipeline: clean JD → retrieve experience → call Claude → return structured analysis.
    """
    # 1. Clean the JD: separate core responsibilities from qualifications boilerplate
    jd_sections = clean_jd(jd_text)
    retrieval_query = get_retrieval_query(jd_text)

    # 2. Retrieve relevant experience chunks
    chunks = retrieve_relevant_chunks(retrieval_query, n_total=n_chunks)

    # 3. Build prompts
    system_prompt = build_system_prompt(rubric_path)
    user_prompt = build_user_prompt(jd_text, chunks, jd_sections=jd_sections)

    # 4. Call Claude with forced tool use
    client = get_client()
    response = _call_anthropic_with_retry(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        on_retry=on_retry,
    )

    # 5. Extract tool use result
    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        raise RuntimeError(
            f"Claude did not call the analyze_job_fit tool. "
            f"Stop reason: {response.stop_reason}. "
            f"Content: {response.content}"
        )

    # 6. Validate with Pydantic
    return JobFitAnalysis(**tool_block.input)


def analyze_jobs_parallel(
    jd_texts: list[str],
    labels: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    n_chunks: int = 8,
) -> list[tuple[str, JobFitAnalysis]]:
    """
    Analyze multiple JDs in parallel using ThreadPoolExecutor.
    Returns list of (label, analysis) tuples.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if labels is None:
        labels = [f"Role {i+1}" for i in range(len(jd_texts))]

    results: list[tuple[str, JobFitAnalysis]] = [None] * len(jd_texts)  # type: ignore

    def _analyze(idx: int, label: str, jd: str):
        analysis = analyze_job(jd, model=model, n_chunks=n_chunks)
        return idx, label, analysis

    with ThreadPoolExecutor(max_workers=min(len(jd_texts), 3)) as executor:
        futures = {
            executor.submit(_analyze, i, label, jd): i
            for i, (label, jd) in enumerate(zip(labels, jd_texts))
        }
        for future in as_completed(futures):
            idx, label, analysis = future.result()
            results[idx] = (label, analysis)

    return results
