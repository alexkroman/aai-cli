"""Shared evaluation helpers for ASR prompt scoring."""

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

_normalize = EnglishTextNormalizer()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate between reference and hypothesis."""
    ref = _normalize(reference or "")
    hyp = _normalize(hypothesis or "")
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    return jiwer.wer(ref, hyp)
