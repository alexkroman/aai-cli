"""Shared service layer — config loading, eval pipeline, and env helpers.

Both CLI commands (cli.py) and agent tools (tools_eval.py) import from here,
avoiding the circular tools → cli dependency.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from .datasets import _resolve_dataset_config, load_all_datasets
from .display import _print_eval_summary, _print_sample_result, default_console
from .eval import compute_laser_score, compute_wer, normalize_text
from .streaming import is_streaming_model
from .transcribe import TranscriptionError, set_api_key, transcribe
from .types import AudioSample, DatasetOptions, EvalSampleResult

CONFIG_PATH = Path(__file__).parent / "conf" / "config.yaml"


def load_config(overrides: dict | None = None, cfg_path: Path | None = None) -> DictConfig:
    """Load config.yaml and merge CLI overrides."""
    base = OmegaConf.load(cfg_path or CONFIG_PATH)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.create(overrides))
    assert isinstance(base, DictConfig)
    return base


def _suppress_loggers() -> None:
    for logger_name in (
        "datasets.info",
        "huggingface_hub",
        "huggingface_hub.repocard",
        "huggingface_hub.utils._headers",
        "httpx",
    ):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def _build_overrides(section: str, **kwargs: object) -> dict[str, dict[str, object]]:
    """Build a config overrides dict from non-None keyword arguments."""
    values = {k: v for k, v in kwargs.items() if v is not None}
    if not values:
        return {}
    return {section: values}


def get_env_key(name: str) -> str:
    """Read a required environment variable or raise ValueError."""
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"Missing env var: {name}")
    return value


def setup_config(
    section: str,
    ds_opts: DatasetOptions | None = None,
    cfg_path: Path | None = None,
    **overrides_kwargs,
) -> DictConfig:
    """Common setup: suppress loggers, build overrides, load config, resolve datasets."""
    _suppress_loggers()
    overrides = _build_overrides(section, **overrides_kwargs)
    cfg = load_config(overrides or None, cfg_path=cfg_path)
    return _resolve_dataset_config(cfg, ds_opts or DatasetOptions())


def run_eval(
    cfg: DictConfig,
    aai_key: str,
    hf_token: str,
    laser: bool = False,
    console: Console | None = None,
) -> None:
    """Run evaluation: transcribe samples and report TTFB, TTFS, and WER."""
    console = console or default_console
    set_api_key(aai_key)
    console.print("Loading datasets...")
    samples = load_all_datasets(cfg, hf_token, total_samples=cfg.eval.max_samples, console=console)

    prompt = cfg.eval.prompt
    num_threads = cfg.eval.num_threads
    speech_model = cfg.eval.speech_model
    streaming = is_streaming_model(speech_model)
    api_mode = "streaming" if streaming else "batch"
    api_host = cfg.eval.get("api_host", None)

    console.print(f'[bold]Evaluating with prompt:[/bold] "{prompt}"')
    metrics_label = "WER + LASER" if laser else "WER"
    console.print(
        f"[dim]Model: {speech_model} ({api_mode}), Samples: {len(samples)}, "
        f"Threads: {num_threads}, Metrics: {metrics_label}[/dim]"
    )
    if api_host:
        console.print(f"[dim]API host: {api_host}[/dim]")

    def _transcribe(sample: AudioSample) -> EvalSampleResult:
        try:
            result = transcribe(
                sample.audio, prompt, aai_key, speech_model=speech_model, api_host=api_host
            )
        except TranscriptionError as e:
            raise RuntimeError(str(e)) from e

        norm_ref = normalize_text(sample.reference)
        norm_hyp = normalize_text(result.text)
        wer = compute_wer(norm_ref, norm_hyp, pre_normalized=True)
        laser_result = None
        if laser:
            laser_result = compute_laser_score(norm_ref, norm_hyp, pre_normalized=True)
        return EvalSampleResult(
            text=result.text,
            wer=wer,
            reference=sample.reference,
            dataset=sample.dataset,
            ttfb=result.ttfb,
            ttfs=result.ttfs,
            laser=laser_result,
        )

    results: list[EvalSampleResult] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(_transcribe, s): i for i, s in enumerate(samples)}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                r = future.result()
            except Exception as exc:
                skipped += 1
                console.print(f"[dim][{i}/{len(samples)}] skipped — error: {exc}[/dim]")
                continue
            results.append(r)
            _print_sample_result(r, i, len(samples), laser, console)

    if skipped:
        console.print(f"[yellow]Skipped {skipped} sample(s) due to errors.[/yellow]")
    if not results:
        console.print("[bold red]All samples failed — no results to summarize.[/bold red]")
        return

    _print_eval_summary(results, len(samples), speech_model, api_mode, laser, console)
