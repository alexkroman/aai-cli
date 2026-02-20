"""Shared evaluation helpers for ASR prompt scoring."""

import json
import logging
import os
import time

import jiwer
import litellm
from pydantic import BaseModel
from whisper_normalizer.english import EnglishTextNormalizer

from .types import LaserResult

_log = logging.getLogger(__name__)

_normalize = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    """Normalize text for ASR evaluation (lowercase, expand contractions, etc.)."""
    return _normalize(text or "")


def compute_wer(reference: str, hypothesis: str, pre_normalized: bool = False) -> float:
    """Compute word error rate between reference and hypothesis."""
    ref = reference if pre_normalized else _normalize(reference or "")
    hyp = hypothesis if pre_normalized else _normalize(hypothesis or "")
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    return jiwer.wer(ref, hyp)


# ---------------------------------------------------------------------------
# LASER Score (LLM-based ASR Evaluation Rubric)
# ---------------------------------------------------------------------------


class LaserResponse(BaseModel):
    word_count: int
    no_penalty_errors: list[str]
    major_errors: list[str]
    minor_errors: list[str]
    total_penalty: float
    laser_score: float


_LASER_SYSTEM_PROMPT = """\
You are an expert linguist evaluating an Automatic Speech Recognition (ASR) system. \
Your task is to compare a Reference (ground truth) with a Hypothesis (ASR prediction). \
Both inputs have already been text-normalized (lowercased, punctuation stripped, numbers \
expanded), so ignore any remaining capitalization or punctuation differences.

Instructions:
1. Align reference and hypothesis word-by-word to find mismatches.
2. Classify every mismatch using the rubric below.
3. Compute: Score = 1 - (Total Penalty / Number of Reference Words). Clamp to [0, 1].

Rubric:

A. No Penalty (0 points) — differences that do not change meaning:
   - Numbers & formatting: "1300" vs "thirteen hundred"
   - Abbreviations: "ATM" vs "A T M", "NBA" vs "N B A"
   - Compound words: "healthcare" vs "health care", "info graphic" vs "infographic"
   - Proper noun spelling variants (same person/place): "Graeme" vs "Graham", \
"Archy" vs "Archie", "Jonathan" vs "Johnathan". \
IMPORTANT: only apply this when both words refer to the same entity. \
Words that look similar but have different meanings are Major errors \
(e.g. "July" vs "slide", "cover" vs "COVID").
   - Contractions & expansions: "going to" vs "gonna", "I am" vs "I'm", \
"All's" vs "All is"
   - Speech disfluencies & fillers removed by ASR: "uh", "um", "ah", false starts \
("c-", "ou-"), and word repetitions ("just just just" → "just") are not errors
   - Discourse markers added or removed at sentence boundaries: "well", "so", "and", \
"but", "you know", "like", "okay" when used as fillers rather than content words. \
"And" or "so" at the start of a sentence is almost always a discourse marker.
   - Singular/plural of the same word: "cancer" vs "cancers", "kid" vs "kids", \
"panelists" vs "panelist", "death" vs "deaths", "item" vs "items", \
"adjustment" vs "adjustments". Always No Penalty — never classify singular/plural \
as Minor or Major.

B. Minor Penalty (0.5 points) — small errors that do not alter core meaning:
   - Single-character spelling errors: "receive" vs "recieve"
   - Minor grammatical markers (tense/number): "He run fast" vs "He runs fast"
   - Comparative/base form differences: "narrower" vs "narrow", "largest" vs "large"
   - Preposition swaps that preserve meaning: "filter in" vs "filter on", \
"adamant about" vs "adamant on"
   - Minor article omission/addition: "the full commitment" vs "full commitment", \
"a good point" vs "good point"
   - Small proper noun misspellings (1-2 character difference): "Joe" vs "Joel"

C. Major Penalty (1.0 points) — errors that significantly alter meaning:
   - Incorrect word substitutions: "beautiful" vs "ugly", "dispatching" vs "packaging"
   - Meaning-altering spelling errors: "affect" vs "effect", "cover" vs "COVID", \
"that" vs "than", "then" vs "than"
   - Significant omissions or additions of content words. \
Count each omitted or added content word separately.
   - Word reordering that changes intent
   - Output in the wrong language: e.g. "Yeah" transcribed as "Ja" (German)"""

_DEFAULT_LASER_MODEL = "anthropic/claude-haiku-4-5-20251001"

_FIELD_ALIASES = {
    "non_penalizable_errors": "no_penalty_errors",
    "nonpenalizable_errors": "no_penalty_errors",
    "non_penalty_errors": "no_penalty_errors",
    "major_penalizable_errors": "major_errors",
    "minor_penalizable_errors": "minor_errors",
    "score": "laser_score",
}

_LIST_FIELDS = ("no_penalty_errors", "major_errors", "minor_errors")


def _llm_call_with_retry(
    messages: list[dict], model: str, api_key: str | None, max_retries: int = 3
) -> object:
    """Call litellm.completion with exponential-backoff retries."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return litellm.completion(
                model=model,
                messages=messages,
                api_key=api_key,
                max_tokens=2048,
                temperature=0.0,
                response_format=LaserResponse,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                _log.debug("LASER LLM call failed (attempt %d/%d): %s", attempt, max_retries, exc)
                time.sleep(2**attempt)
    raise RuntimeError(
        f"LASER scoring failed after {max_retries} attempts: {last_exc}"
    ) from last_exc


def _normalize_laser_response(content: str | None) -> LaserResponse:
    """Parse, alias-map, and validate the raw LLM response into a LaserResponse."""
    if content is None:
        raise RuntimeError("LASER LLM returned empty content")
    raw = json.loads(content) if isinstance(content, str) else content
    if isinstance(raw, dict):
        raw = {_FIELD_ALIASES.get(k, k): v for k, v in raw.items()}
        for key in _LIST_FIELDS:
            val = raw.get(key)
            if isinstance(val, str):
                try:
                    raw[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    raw[key] = [val] if val else []
    parsed = LaserResponse.model_validate(raw)
    parsed.laser_score = max(0.0, min(1.0, parsed.laser_score))
    return parsed


def _laser_response_to_result(parsed: LaserResponse) -> LaserResult:
    """Convert a validated LaserResponse (Pydantic) to a LaserResult (dataclass)."""
    return LaserResult(
        laser_score=parsed.laser_score,
        word_count=parsed.word_count,
        no_penalty_errors=parsed.no_penalty_errors,
        major_errors=parsed.major_errors,
        minor_errors=parsed.minor_errors,
        total_penalty=parsed.total_penalty,
    )


def compute_laser_score(
    reference: str,
    hypothesis: str,
    model: str = _DEFAULT_LASER_MODEL,
    api_key: str | None = None,
    pre_normalized: bool = False,
) -> LaserResult:
    """Compute LASER score for a reference/hypothesis pair using an LLM.

    Returns a LaserResult with laser_score (0-1, 1 = perfect), word_count,
    error lists, and total_penalty.
    """
    ref = reference if pre_normalized else _normalize(reference or "")
    hyp = hypothesis if pre_normalized else _normalize(hypothesis or "")

    if not ref:
        return LaserResult(
            word_count=0,
            no_penalty_errors=[],
            major_errors=[],
            minor_errors=[],
            total_penalty=0.0,
            laser_score=1.0,
        )
    if not hyp:
        wc = len(ref.split())
        return LaserResult(
            word_count=wc,
            no_penalty_errors=[],
            major_errors=[f"entire sentence missing ({wc} words)"],
            minor_errors=[],
            total_penalty=float(wc),
            laser_score=0.0,
        )

    messages = [
        {"role": "system", "content": _LASER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Reference: {ref}\nHypothesis: {hyp}"},
    ]
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    response = _llm_call_with_retry(messages, model, resolved_key)
    parsed = _normalize_laser_response(response.choices[0].message.content)
    return _laser_response_to_result(parsed)
