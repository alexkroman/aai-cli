"""Agent tools for evaluation and optimization."""

import os

from smolagents import tool

from .tools import _make_capture_console


@tool
def eval_prompt(
    prompt: str,
    max_samples: int = 10,
    num_threads: int = 12,
    speech_model: str = "universal-3-pro",
    laser: bool = False,
    dataset: str = "all",
    hf_dataset: str = "",
    hf_config: str = "default",
    audio_column: str = "audio",
    text_column: str = "text",
    split: str = "test",
) -> str:
    """Evaluate an AssemblyAI transcription prompt and measure Word Error Rate (WER).

    Sends audio samples to AssemblyAI for transcription using the given prompt,
    then compares the output against ground-truth references. Reports per-sample
    results and overall WER. Lower WER = better transcription quality.

    Use this when a user asks to evaluate, test, or benchmark AssemblyAI transcription,
    measure WER, or compare prompt performance.

    IMPORTANT: batch and streaming models use DIFFERENT APIs and are NOT interchangeable.
    "u3-pro" is NOT shorthand for "universal-3-pro".
    Batch model (uploads audio, gets results back):
    - "universal-3-pro" (default)
    Streaming models (real-time WebSocket):
    - "u3-pro"
    - "universal-streaming-english"
    - "universal-streaming-multilingual"
    When unsure which model the user wants, ask them to clarify batch vs streaming.

    Args:
        prompt: The transcription prompt to evaluate (e.g. "Transcribe verbatim.").
        max_samples: Number of audio samples to transcribe (default: 10 for quick test, use 50+ for reliable results).
        num_threads: Parallel transcription threads (default: 12).
        speech_model: Speech model to use. Batch: "universal-3-pro" (default). Streaming: "u3-pro", "universal-streaming-english", "universal-streaming-multilingual".
        laser: Compute LASER score (LLM-based rubric) alongside WER (default: False).
        dataset: Preconfigured dataset shortcut. One of "earnings22", "peoples", "ami", "loquacious", "gigaspeech", "tedlium", "commonvoice", "librispeech", "librispeech-other", or "all" (default). Ignored when hf_dataset is set.
        hf_dataset: Any HF audio dataset path (e.g. "mozilla-foundation/common_voice_11_0"). Overrides dataset.
        hf_config: HF dataset config/subset name (default: "default").
        audio_column: Name of the audio column in the dataset (default: "audio").
        text_column: Name of the text/reference column in the dataset (default: "text").
        split: Dataset split to use (default: "test").
    """
    from .service import run_eval, setup_config
    from .types import DatasetOptions

    ds_opts = DatasetOptions(
        hf_dataset=hf_dataset or None,
        hf_config=hf_config,
        audio_column=audio_column,
        text_column=text_column,
        split=split,
        dataset=dataset or None,
    )
    cfg = setup_config(
        "eval",
        ds_opts=ds_opts,
        prompt=prompt,
        max_samples=max_samples,
        num_threads=num_threads,
        speech_model=speech_model,
    )
    aai_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    console, get_output = _make_capture_console()
    run_eval(cfg, aai_key, hf_token, laser=laser, console=console)
    return get_output() or "(no output)"


@tool
def optimize_prompt(
    starting_prompt: str,
    iterations: int = 5,
    samples: int = 50,
    candidates_per_step: int = 1,
    num_threads: int = 12,
    llm_model: str = "",
    laser: bool = False,
    dataset: str = "all",
    hf_dataset: str = "",
    hf_config: str = "default",
    audio_column: str = "audio",
    text_column: str = "text",
    split: str = "test",
) -> str:
    """Optimize an AssemblyAI transcription prompt using OPRO (LLM-guided optimization).

    Iteratively proposes improved transcription prompts by analyzing transcription
    errors and using an LLM to suggest fixes. Each iteration: generates candidate
    prompts, evaluates them against audio datasets, and keeps the best.

    This is long-running — use fewer iterations/samples for quick experiments.

    Use this when a user asks to optimize, improve, or tune a transcription prompt.

    Args:
        starting_prompt: The seed prompt to begin optimizing from.
        iterations: Number of optimization rounds (default: 5).
        samples: Total audio samples to evaluate per candidate (default: 50, minimum: 5, 50+ recommended).
        candidates_per_step: Number of candidate prompts generated per iteration (default: 1).
        num_threads: Parallel transcription threads (default: 12).
        llm_model: Model used to generate candidate prompts. Leave empty to use config default.
        laser: Optimize using LASER metric instead of WER (default: False).
        dataset: Preconfigured dataset shortcut. One of "earnings22", "peoples", "ami", "loquacious", "gigaspeech", "tedlium", "commonvoice", "librispeech", "librispeech-other", or "all" (default). Ignored when hf_dataset is set.
        hf_dataset: Any HF audio dataset path (e.g. "mozilla-foundation/common_voice_11_0"). Overrides dataset.
        hf_config: HF dataset config/subset name (default: "default").
        audio_column: Name of the audio column in the dataset (default: "audio").
        text_column: Name of the text/reference column in the dataset (default: "text").
        split: Dataset split to use (default: "test").
    """
    # --- Validation ---
    if samples < 5:
        return "Error: samples must be at least 5 (50+ recommended for reliable results)."
    if iterations < 1:
        return "Error: iterations must be at least 1."
    if candidates_per_step < 1:
        return "Error: candidates_per_step must be at least 1."
    if not starting_prompt.strip():
        return "Error: starting_prompt cannot be empty."

    from .datasets import load_all_datasets
    from .optimizer import run_optimization
    from .service import setup_config
    from .types import DatasetOptions

    ds_opts = DatasetOptions(
        hf_dataset=hf_dataset or None,
        hf_config=hf_config,
        audio_column=audio_column,
        text_column=text_column,
        split=split,
        dataset=dataset or None,
    )
    cfg = setup_config(
        "optimization",
        ds_opts=ds_opts,
        starting_prompt=starting_prompt,
        iterations=iterations,
        samples=samples,
        candidates_per_step=candidates_per_step,
        num_threads=num_threads,
        llm_model=llm_model or None,
    )
    aai_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    console, get_output = _make_capture_console()
    eval_samples = load_all_datasets(cfg, hf_token, console=console)
    result = run_optimization(
        eval_samples=eval_samples,
        api_key=aai_key,
        starting_prompt=cfg.optimization.starting_prompt,
        iterations=cfg.optimization.iterations,
        console=console,
        candidates_per_step=cfg.optimization.candidates_per_step,
        llm_model=cfg.optimization.llm_model,
        num_threads=cfg.optimization.num_threads,
        laser=laser,
    )

    lines = [get_output()]
    lines.append(f"\nBest Prompt: {result.best_prompt}")
    lines.append(f"Best Score: {result.best_score}")
    return "\n".join(lines) or "(no output)"
