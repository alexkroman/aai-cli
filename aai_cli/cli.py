"""Entry point — Hydra-based declarative config."""

import json
import logging
import os
import sys
from pathlib import Path

import hydra
import pyfiglet
from datasets import Audio, load_dataset
from huggingface_hub import login as hf_login
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from .optimizer import _evaluate_prompt, _mean_wer, run_optimization

THEME = Theme(
    {
        "banner": "bold cyan",
        "border": "bright_blue",
        "heading": "bold bright_blue",
        "progress": "cyan",
        "success": "green",
        "warning": "yellow",
        "muted": "dim",
    }
)

console = Console(theme=THEME)

SUBTITLE = "Prompt Optimizer for Universal-3-Pro"


def print_banner():
    figlet_text = pyfiglet.figlet_format("AAI", font="slant")
    banner = figlet_text.rstrip() + "\n\n" + SUBTITLE
    console.print(
        Panel(
            Text(banner, style="banner"),
            border_style="border",
            padding=(0, 2),
        )
    )


def get_env_key(name: str) -> str:
    """Read a required environment variable or exit with a clear error."""
    value = os.environ.get(name, "")
    if not value:
        console.print(f"  [warning]Missing env var: {name}[/warning]")
        sys.exit(1)
    return value


def load_dataset_samples(
    ds_name: str, ds_cfg: DictConfig, num_samples: int, hf_token: str
) -> list[dict]:
    """Load samples from one HF dataset."""
    console.print(f"  [progress]Loading {ds_name}...[/progress]")
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

    console.print(f"  [success]Loaded {len(samples)} samples from {ds_name}[/success]")
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
        all_samples.extend(samples)

    console.print(
        f"  [success]Total: {len(all_samples)} samples from {len(ds_names)} datasets[/success]"
    )
    return all_samples


def run_eval(cfg: DictConfig, aai_key: str, hf_token: str) -> None:
    """Run evaluation: transcribe samples, print ref/hyp pairs, and report WER."""
    console.print("\n  [heading]Loading Datasets[/heading]")
    samples = load_all_datasets(cfg, hf_token, total_samples=cfg.eval.max_samples)

    prompt = cfg.eval.prompt
    num_threads = cfg.eval.num_threads
    console.print(f'\n  [heading]Evaluating with prompt:[/heading] "{prompt}"')
    console.print(f"  [muted]Samples: {len(samples)}, Threads: {num_threads}[/muted]\n")

    results = _evaluate_prompt(prompt, samples, aai_key, num_threads, desc="Eval")

    console.print("\n  [heading]Results[/heading]\n")
    for i, r in enumerate(results, 1):
        wer_pct = r["wer"] * 100
        if r["wer"] == 0.0:
            wer_style = "success"
        elif r["wer"] < 0.1:
            wer_style = "progress"
        else:
            wer_style = "warning"

        console.print(
            f"  [bold]--- Sample {i}/{len(results)} (WER: [{wer_style}]{wer_pct:.1f}%[/{wer_style}]) ---[/bold]"
        )
        console.print(f"  [bold]REF:[/bold] {r['reference']}")
        console.print(f"  [bold]HYP:[/bold] {r['hypothesis']}")
        console.print()

    overall_wer = _mean_wer(results)
    console.print(
        Panel(
            f'  WER: [bold]{overall_wer * 100:.2f}%[/bold]  |  Samples: {len(results)}  |  Prompt: "{prompt}"',
            title="[bold]Evaluation Summary[/bold]",
            border_style="border",
            padding=(1, 2),
        )
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """AAI CLI — optimize or evaluate AssemblyAI universal-3-pro prompts."""
    for logger_name in (
        "datasets.info",
        "huggingface_hub",
        "huggingface_hub.repocard",
        "huggingface_hub.utils._headers",
        "httpx",
    ):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    print_banner()

    # API keys from env vars
    console.print("  [heading]API Keys[/heading]")
    aai_key = get_env_key("ASSEMBLYAI_API_KEY")
    hf_token = get_env_key("HF_TOKEN")
    console.print("  [success]All keys loaded from environment[/success]")

    mode = cfg.get("mode", "optimize")

    if mode == "eval":
        run_eval(cfg, aai_key, hf_token)
        return

    # --- optimize mode ---

    # Check for previous state to resume from
    state_file = Path("outputs") / "optimization_state.json"
    resume_history = None
    if state_file.exists():
        prev_state = json.loads(state_file.read_text())
        if "history" in prev_state and prev_state["history"]:
            resume_history = prev_state["history"]
            console.print(
                f"\n  [heading]Resuming from {state_file} "
                f"({len(resume_history)} previous entries)[/heading]"
            )

    # Load data
    console.print("\n  [heading]Loading Datasets[/heading]")
    eval_samples = load_all_datasets(cfg, hf_token)

    # Run optimization
    result = run_optimization(
        eval_samples=eval_samples,
        api_key=aai_key,
        starting_prompt=cfg.optimization.starting_prompt,
        num_threads=cfg.optimization.num_threads,
        iterations=cfg.optimization.iterations,
        console=console,
        candidates_per_step=cfg.optimization.candidates_per_step,
        trajectory_size=cfg.optimization.trajectory_size,
        llm_model=cfg.optimization.llm_model,
        resume_history=resume_history,
        seed=cfg.optimization.seed,
        meta_every=cfg.optimization.meta_every,
    )

    # Add dataset info to result
    result["datasets"] = list(cfg.datasets.keys())

    # Display result
    best = result["best_prompt"]
    console.print()
    console.print("  [bold success]Best prompt:[/bold success]")
    console.print(f"\n{best}\n")

    # Save
    output_path = Path("outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    state_file = output_path / "optimization_state.json"
    state_file.write_text(json.dumps(result, indent=2))
    console.print(f"\n  [muted]Saved to {state_file}[/muted]\n")


def app():
    """Entry point for `[project.scripts]`."""
    main()


if __name__ == "__main__":
    app()
