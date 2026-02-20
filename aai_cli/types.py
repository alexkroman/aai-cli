from dataclasses import dataclass, field
from typing import Any

AudioData = dict[str, Any] | Any
"""TypeAlias for audio data (HF dict with bytes/array/path, or duck-typed object)."""


@dataclass
class TranscriptionResult:
    text: str
    ttfb: float | None = None
    ttfs: float | None = None


@dataclass
class AudioSample:
    audio: AudioData
    reference: str
    dataset: str = ""


@dataclass
class DatasetOptions:
    """Dataset selection parameters shared across CLI commands and tool functions."""

    hf_dataset: str | None = None
    hf_config: str = "default"
    audio_column: str = "audio"
    text_column: str = "text"
    split: str = "test"
    dataset: str | None = None


@dataclass
class LaserResult:
    """Typed result from compute_laser_score (replaces raw dict)."""

    laser_score: float
    word_count: int
    no_penalty_errors: list[str]
    major_errors: list[str]
    minor_errors: list[str]
    total_penalty: float


@dataclass
class EvalSampleResult:
    """Per-sample evaluation result flowing through service.py -> display.py."""

    text: str
    wer: float
    reference: str
    dataset: str = ""
    ttfb: float | None = None
    ttfs: float | None = None
    laser: LaserResult | None = None


@dataclass
class OptimizationResult:
    """Result from run_optimization (replaces raw dict)."""

    best_prompt: str
    best_score: float
    metric: str
    starting_prompt: str
    model: str
    total_eval_samples: int
    optimizer: str
    num_trials: int
    llm_model: str
    timestamp: str
    datasets: list[str] = field(default_factory=list)
