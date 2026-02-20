"""Entry point — routes to agent (default) or Typer optimize/eval subcommands."""

import dataclasses
import json
import sys
from pathlib import Path

import typer

from .datasets import _validate_dataset_args, load_all_datasets
from .display import default_console
from .optimizer import run_optimization
from .service import get_env_key, run_eval, setup_config
from .types import DatasetOptions

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
    laser: bool = typer.Option(False, "--laser", help="Optimize using LASER metric instead of WER"),
):
    """Run DSPy optimization."""
    _validate_dataset_args(hf_dataset, dataset)
    ds_opts = DatasetOptions(
        hf_dataset=hf_dataset,
        hf_config=hf_config or "default",
        audio_column=audio_column or "audio",
        text_column=text_column or "text",
        split=split or "test",
        dataset=dataset,
    )
    cfg = setup_config(
        "optimization",
        ds_opts=ds_opts,
        cfg_path=config,
        samples=samples,
        iterations=iterations,
        starting_prompt=starting_prompt,
        candidates_per_step=candidates_per_step,
        num_threads=num_threads,
        llm_model=llm_model,
    )
    aai_key = get_env_key("ASSEMBLYAI_API_KEY")
    hf_token = get_env_key("HF_TOKEN")

    default_console.print("Loading datasets...")
    eval_samples = load_all_datasets(cfg, hf_token)

    result = run_optimization(
        eval_samples=eval_samples,
        api_key=aai_key,
        starting_prompt=cfg.optimization.starting_prompt,
        iterations=cfg.optimization.iterations,
        console=default_console,
        candidates_per_step=cfg.optimization.candidates_per_step,
        llm_model=cfg.optimization.llm_model,
        num_threads=cfg.optimization.num_threads,
        laser=laser,
    )

    result.datasets = list(cfg.datasets.keys())

    best = result.best_prompt
    default_console.print()
    default_console.print(f"[bold]Best Prompt:[/bold] {best}")

    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dataclasses.asdict(result), indent=2))
        default_console.print(f"[dim]Saved to {p}[/dim]")
    default_console.print()


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
    laser: bool = typer.Option(
        False, "--laser", help="Compute LASER score (LLM-based rubric) alongside WER"
    ),
):
    """Evaluate a transcription prompt (WER, and optionally LASER)."""
    _validate_dataset_args(hf_dataset, dataset)
    ds_opts = DatasetOptions(
        hf_dataset=hf_dataset,
        hf_config=hf_config or "default",
        audio_column=audio_column or "audio",
        text_column=text_column or "text",
        split=split or "test",
        dataset=dataset,
    )
    cfg = setup_config(
        "eval",
        ds_opts=ds_opts,
        cfg_path=config,
        prompt=prompt,
        max_samples=max_samples,
        num_threads=num_threads,
        speech_model=speech_model,
        api_host=api_host,
    )
    aai_key = get_env_key("ASSEMBLYAI_API_KEY")
    hf_token = get_env_key("HF_TOKEN")

    run_eval(cfg, aai_key, hf_token, laser=laser)


def launch_agent(extra_args: list[str]) -> None:
    """Launch the built-in coding agent."""
    from .repl import run_agent

    run_agent(extra_args or None)


def app() -> None:
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
