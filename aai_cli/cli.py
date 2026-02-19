"""Entry point — routes to agent (default) or Typer optimize/eval subcommands."""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from datasets import Audio, load_dataset
from huggingface_hub import login as hf_login
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from .eval import compute_wer
from .optimizer import run_optimization
from .streaming import is_streaming_model, transcribe_streaming
from .transcribe import transcribe_assemblyai

console = Console()

CONFIG_PATH = Path(__file__).parent / "conf" / "config.yaml"


def load_config(overrides: dict | None = None, cfg_path: Path | None = None) -> DictConfig:
    """Load config.yaml and merge CLI overrides."""
    base = OmegaConf.load(cfg_path or CONFIG_PATH)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.create(overrides))
    return base


def _suppress_loggers():
    for logger_name in (
        "datasets.info",
        "huggingface_hub",
        "huggingface_hub.repocard",
        "huggingface_hub.utils._headers",
        "httpx",
    ):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def _filter_datasets(cfg, dataset):
    """Keep only named dataset, or all if 'all'/None."""
    if not dataset or dataset == "all":
        return cfg
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    container["datasets"] = {dataset: container["datasets"][dataset]}
    return OmegaConf.create(container)


def _apply_hf_dataset_override(
    cfg: DictConfig,
    hf_dataset: str,
    hf_config: str = "default",
    audio_column: str = "audio",
    text_column: str = "text",
    split: str = "test",
) -> DictConfig:
    """Replace cfg.datasets with a single ad-hoc HF dataset entry."""
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    container["datasets"] = {
        "custom": {
            "path": hf_dataset,
            "config": hf_config,
            "audio_column": audio_column,
            "text_column": text_column,
            "split": split,
        }
    }
    return OmegaConf.create(container)


def print_banner():
    console.print("Welcome to the AssemblyAI agent 🌟 ✨\n")


def get_env_key(name: str) -> str:
    """Read a required environment variable or exit with a clear error."""
    value = os.environ.get(name, "")
    if not value:
        console.print(f"  Missing env var: {name}", style="bold red")
        sys.exit(1)
    return value


def load_dataset_samples(
    ds_name: str, ds_cfg: DictConfig, num_samples: int, hf_token: str
) -> list[dict]:
    """Load samples from one HF dataset."""
    console.print(f"Loading [italic]{ds_name}[/italic]...")
    hf_login(token=hf_token, add_to_git_credential=False)

    kwargs: dict = {
        "path": ds_cfg.path,
        "split": ds_cfg.split,
        "streaming": True,
        "token": hf_token,
    }
    if ds_cfg.config:
        kwargs["name"] = ds_cfg.config
    dataset = load_dataset(**kwargs)
    dataset = dataset.cast_column(ds_cfg.audio_column, Audio(decode=False))

    audio_col = ds_cfg.audio_column
    text_col = ds_cfg.text_column

    samples: list[dict] = []
    for sample in dataset:
        reference = sample[text_col]
        if isinstance(reference, str) and reference.strip() == "ignore_time_segment_in_scoring":
            continue
        if isinstance(reference, str) and "inaudible" in reference.lower():
            continue
        samples.append({"audio": sample[audio_col], "reference": reference})
        if len(samples) >= num_samples:
            break

    console.print(f"Loaded [bold]{len(samples)}[/bold] samples from [italic]{ds_name}[/italic]")
    return samples


def load_all_datasets(
    cfg: DictConfig, hf_token: str, total_samples: int | None = None
) -> list[dict]:
    """Split total samples evenly across configured datasets and collect them."""
    ds_names = list(cfg.datasets.keys())
    total = total_samples if total_samples is not None else cfg.optimization.samples
    per_ds = total // len(ds_names)
    remainder = total % len(ds_names)

    all_samples: list[dict] = []
    for i, name in enumerate(ds_names):
        n = per_ds + (1 if i < remainder else 0)
        samples = load_dataset_samples(name, cfg.datasets[name], n, hf_token)
        for s in samples:
            s["dataset"] = name
        all_samples.extend(samples)

    console.print(
        f"Total: [bold]{len(all_samples)}[/bold] samples from [bold]{len(ds_names)}[/bold] datasets"
    )
    return all_samples


def _latency_stats(values: list[float]) -> str:
    """Format avg/P95/min/max for a list of latency values."""
    s = sorted(values)
    avg = sum(s) / len(s)
    p95_idx = min(int(len(s) * 0.95), len(s) - 1)
    return f"avg {avg:.2f}s | P95 {s[p95_idx]:.2f}s | Min {s[0]:.2f}s | Max {s[-1]:.2f}s"


def run_eval(cfg: DictConfig, aai_key: str, hf_token: str) -> None:
    """Run evaluation: transcribe samples and report TTFB, TTFS, and WER."""
    console.print("Loading datasets...")
    samples = load_all_datasets(cfg, hf_token, total_samples=cfg.eval.max_samples)

    prompt = cfg.eval.prompt
    num_threads = cfg.eval.num_threads
    speech_model = cfg.eval.speech_model
    streaming = is_streaming_model(speech_model)
    api_mode = "streaming" if streaming else "batch"
    api_host = cfg.eval.get("api_host", None)

    console.print(f'[bold]Evaluating with prompt:[/bold] "{prompt}"')
    console.print(
        f"[dim]Model: {speech_model} ({api_mode}), Samples: {len(samples)}, Threads: {num_threads}[/dim]"
    )
    if api_host:
        console.print(f"[dim]API host: {api_host}[/dim]")

    def _transcribe(sample: dict) -> dict:
        if streaming:
            result = transcribe_streaming(
                sample["audio"], prompt, aai_key, speech_model=speech_model, api_host=api_host
            )
        else:
            result = transcribe_assemblyai(sample["audio"], prompt, aai_key)
        wer = compute_wer(sample["reference"], result["text"])
        return {
            **result,
            "wer": wer,
            "reference": sample["reference"],
            "dataset": sample.get("dataset", ""),
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(_transcribe, s): i for i, s in enumerate(samples)}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            wer_pct = r["wer"] * 100.0
            latency_parts = []
            if r["ttfb"] is not None:
                latency_parts.append(f"TTFB {r['ttfb']:.2f}s")
            if r["ttfs"] is not None:
                latency_parts.append(f"TTFS {r['ttfs']:.2f}s")
            latency_str = " | ".join(latency_parts) if latency_parts else ""
            ds_tag = f" ({r['dataset']})" if r["dataset"] else ""
            console.print(
                f"[dim][{i}/{len(samples)}]{ds_tag} WER: {wer_pct:.1f}% | {latency_str}[/dim]"
            )
            console.print(f"  [green]REF:[/green] {r['reference']}")
            console.print(f"  [yellow]HYP:[/yellow] {r['text']}")

    wer_values = [r["wer"] for r in results]
    avg_wer = sum(wer_values) / len(wer_values) * 100.0

    ttfb_values = [r["ttfb"] for r in results if r["ttfb"] is not None]
    ttfs_values = [r["ttfs"] for r in results if r["ttfs"] is not None]

    console.print()
    if ttfb_values:
        console.print(f"[bold]TTFB:[/bold] {_latency_stats(ttfb_values)}")
    if ttfs_values:
        console.print(f"[bold]TTFS:[/bold] {_latency_stats(ttfs_values)}")

    # Per-dataset WER breakdown when multiple datasets are used
    datasets_seen = sorted({r["dataset"] for r in results if r["dataset"]})
    if len(datasets_seen) > 1:
        for ds_name in datasets_seen:
            ds_wers = [r["wer"] for r in results if r["dataset"] == ds_name]
            ds_avg = sum(ds_wers) / len(ds_wers) * 100.0
            console.print(f"[bold]WER ({ds_name}): {ds_avg:.2f}%[/bold] | Samples: {len(ds_wers)}")

    console.print(
        f"[bold]WER: {avg_wer:.2f}%[/bold] | Model: {speech_model} ({api_mode}) | Samples: {len(samples)}"
    )


# ---------------------------------------------------------------------------
# Typer subcommands
# ---------------------------------------------------------------------------

typer_app = typer.Typer(invoke_without_command=True, no_args_is_help=False)


@typer_app.command("optimize")
def optimize_cmd(
    samples: int | None = typer.Option(None, help="Total audio samples"),
    iterations: int | None = typer.Option(None, help="Optimization rounds"),
    starting_prompt: str | None = typer.Option(None, help="Seed prompt"),
    candidates_per_step: int | None = typer.Option(None, help="Candidates per iteration"),
    num_threads: int | None = typer.Option(None, help="Parallel threads"),
    llm_model: str | None = typer.Option(None, help="LLM for candidate generation"),
    output: str | None = typer.Option(None, help="Output JSON path"),
    dataset: str | None = typer.Option(None, help="Dataset name or 'all'"),
    hf_dataset: str | None = typer.Option(
        None, help="HF dataset path (e.g. mozilla-foundation/common_voice_11_0)"
    ),
    hf_config: str | None = typer.Option("default", help="HF dataset config/subset"),
    audio_column: str | None = typer.Option("audio", help="Audio column name"),
    text_column: str | None = typer.Option("text", help="Text/reference column name"),
    split: str | None = typer.Option("test", help="Dataset split"),
    config: Path | None = typer.Option(None, help="Alternate config YAML"),
):
    """Run OPRO prompt optimization."""
    _suppress_loggers()

    if hf_dataset and dataset:
        console.print("--hf-dataset and --dataset are mutually exclusive.", style="bold red")
        raise typer.Exit(code=1)

    overrides: dict = {}
    if samples is not None:
        overrides.setdefault("optimization", {})["samples"] = samples
    if iterations is not None:
        overrides.setdefault("optimization", {})["iterations"] = iterations
    if starting_prompt is not None:
        overrides.setdefault("optimization", {})["starting_prompt"] = starting_prompt
    if candidates_per_step is not None:
        overrides.setdefault("optimization", {})["candidates_per_step"] = candidates_per_step
    if num_threads is not None:
        overrides.setdefault("optimization", {})["num_threads"] = num_threads
    if llm_model is not None:
        overrides.setdefault("optimization", {})["llm_model"] = llm_model

    cfg = load_config(overrides or None, cfg_path=config)
    if hf_dataset:
        cfg = _apply_hf_dataset_override(
            cfg, hf_dataset, hf_config, audio_column, text_column, split
        )
    else:
        cfg = _filter_datasets(cfg, dataset)

    aai_key = get_env_key("ASSEMBLYAI_API_KEY")
    hf_token = get_env_key("HF_TOKEN")

    console.print("Loading datasets...")
    eval_samples = load_all_datasets(cfg, hf_token)

    result = run_optimization(
        eval_samples=eval_samples,
        api_key=aai_key,
        starting_prompt=cfg.optimization.starting_prompt,
        iterations=cfg.optimization.iterations,
        console=console,
        candidates_per_step=cfg.optimization.candidates_per_step,
        llm_model=cfg.optimization.llm_model,
        num_threads=cfg.optimization.num_threads,
        output_path=output,
    )

    result["datasets"] = list(cfg.datasets.keys())

    best = result["best_prompt"]
    console.print()
    console.print(f"[bold]Best Prompt:[/bold] {best}")

    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))
        console.print(f"[dim]Saved to {p}[/dim]")
    console.print()


@typer_app.command("eval")
def eval_cmd(
    prompt: str | None = typer.Option(None, help="Transcription prompt"),
    max_samples: int | None = typer.Option(None, help="Number of samples"),
    num_threads: int | None = typer.Option(None, help="Parallel threads"),
    speech_model: str | None = typer.Option(
        None,
        help="Speech model: 'universal-3-pro' (batch) or 'u3-pro'/'universal-streaming-english'/'universal-streaming-multilingual' (streaming)",
    ),
    api_host: str | None = typer.Option(
        None,
        help="Streaming API host override (e.g. streaming.edge.assemblyai.com)",
    ),
    dataset: str | None = typer.Option(None, help="Dataset name or 'all'"),
    hf_dataset: str | None = typer.Option(
        None, help="HF dataset path (e.g. mozilla-foundation/common_voice_11_0)"
    ),
    hf_config: str | None = typer.Option("default", help="HF dataset config/subset"),
    audio_column: str | None = typer.Option("audio", help="Audio column name"),
    text_column: str | None = typer.Option("text", help="Text/reference column name"),
    split: str | None = typer.Option("test", help="Dataset split"),
    config: Path | None = typer.Option(None, help="Alternate config YAML"),
):
    """Evaluate a transcription prompt (WER)."""
    _suppress_loggers()

    if hf_dataset and dataset:
        console.print("--hf-dataset and --dataset are mutually exclusive.", style="bold red")
        raise typer.Exit(code=1)

    overrides: dict = {}
    if prompt is not None:
        overrides.setdefault("eval", {})["prompt"] = prompt
    if max_samples is not None:
        overrides.setdefault("eval", {})["max_samples"] = max_samples
    if num_threads is not None:
        overrides.setdefault("eval", {})["num_threads"] = num_threads
    if speech_model is not None:
        overrides.setdefault("eval", {})["speech_model"] = speech_model
    if api_host is not None:
        overrides.setdefault("eval", {})["api_host"] = api_host

    cfg = load_config(overrides or None, cfg_path=config)
    if hf_dataset:
        cfg = _apply_hf_dataset_override(
            cfg, hf_dataset, hf_config, audio_column, text_column, split
        )
    else:
        cfg = _filter_datasets(cfg, dataset)

    aai_key = get_env_key("ASSEMBLYAI_API_KEY")
    hf_token = get_env_key("HF_TOKEN")

    run_eval(cfg, aai_key, hf_token)


def launch_agent(extra_args: list[str]) -> None:
    """Launch the built-in coding agent."""
    from .repl import run_agent

    run_agent(extra_args or None)


def app():
    """Entry point for `[project.scripts]`.

    Routes:
      aai                               → launch coding agent (default)
      aai optimize [--flag value ...]   → prompt optimization
      aai eval [--flag value ...]       → evaluation
    """
    args = sys.argv[1:]

    if args and args[0] in ("optimize", "eval"):
        typer_app()
    else:
        launch_agent(args)


if __name__ == "__main__":
    app()
