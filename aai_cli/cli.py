"""Entry point — routes to aider (default) or Hydra optimize/eval subcommands."""

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

from .optimizer import _evaluate_prompt, _mean_wer, run_optimization

console = Console()


def print_banner():
    figlet_text = pyfiglet.figlet_format("AAI", font="slant")
    banner = figlet_text.rstrip()
    console.print(Panel(banner, padding=(0, 2)))


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
        all_samples.extend(samples)

    console.print(
        f"Total: [bold]{len(all_samples)}[/bold] samples from [bold]{len(ds_names)}[/bold] datasets"
    )
    return all_samples


def run_eval(cfg: DictConfig, aai_key: str, hf_token: str) -> None:
    """Run evaluation: transcribe samples, print ref/hyp pairs, and report WER."""
    console.print("Loading datasets...")
    samples = load_all_datasets(cfg, hf_token, total_samples=cfg.eval.max_samples)

    prompt = cfg.eval.prompt
    num_threads = cfg.eval.num_threads
    console.print(f'[bold]Evaluating with prompt:[/bold] "{prompt}"')
    console.print(f"[dim]Samples: {len(samples)}, Threads: {num_threads}[/dim]")

    results = _evaluate_prompt(prompt, samples, aai_key, num_threads, desc="Eval")

    console.print()
    for i, r in enumerate(results, 1):
        wer_pct = r["wer"] * 100
        console.print(f"[bold]Sample {i}/{len(results)}[/bold] (WER: {wer_pct:.1f}%)")
        console.print(f"  REF: {r['reference']}")
        console.print(f"  HYP: {r['hypothesis']}")
        console.print()

    overall_wer = _mean_wer(results)
    console.print(
        f'[bold]WER: {overall_wer * 100:.2f}%[/bold] | Samples: {len(results)} | Prompt: "{prompt}"'
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    """AAI CLI — optimize or evaluate AssemblyAI universal-3-pro prompts."""
    for logger_name in (
        "datasets.info",
        "huggingface_hub",
        "huggingface_hub.repocard",
        "huggingface_hub.utils._headers",
        "httpx",
    ):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    aai_key = get_env_key("ASSEMBLYAI_API_KEY")
    hf_token = get_env_key("HF_TOKEN")

    mode = cfg.get("mode", "optimize")

    if mode == "eval":
        run_eval(cfg, aai_key, hf_token)
        return

    # --- optimize mode ---

    # Only resume from previous state when explicitly requested
    resume_history = None
    resume_path = cfg.get("resume", None)
    if resume_path:
        state_file = Path(resume_path)
        if state_file.exists():
            prev_state = json.loads(state_file.read_text())
            if "history" in prev_state and prev_state["history"]:
                resume_history = prev_state["history"]
                console.print(
                    f"[bold]Resuming from {state_file} "
                    f"({len(resume_history)} previous entries)[/bold]"
                )
        else:
            console.print(f"[bold red]Resume file not found: {state_file}[/bold red]")
            sys.exit(1)

    # Load data
    console.print("Loading datasets...")
    eval_samples = load_all_datasets(cfg, hf_token)

    # Determine output path (optional)
    output_file = cfg.get("output", None)

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
        output_path=output_file,
    )

    # Add dataset info to result
    result["datasets"] = list(cfg.datasets.keys())

    # Display result
    best = result["best_prompt"]
    console.print()
    console.print(f"[bold]Best Prompt:[/bold] {best}")

    if output_file:
        p = Path(output_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))
        console.print(f"[dim]Saved to {p}[/dim]")
    console.print()


def launch_agent(extra_args: list[str]) -> None:
    """Launch the built-in coding agent."""
    from .agent import run_agent

    print_banner()
    run_agent(extra_args or None)


def app():
    """Entry point for `[project.scripts]`.

    Routes:
      aai                        → launch coding agent (default)
      aai optimize [hydra-args]  → prompt optimization
      aai eval [hydra-args]      → evaluation
    """
    args = sys.argv[1:]

    if args and args[0] == "optimize":
        sys.argv = [sys.argv[0]] + ["mode=optimize"] + args[1:]
        hydra_main()
    elif args and args[0] == "eval":
        sys.argv = [sys.argv[0]] + ["mode=eval"] + args[1:]
        hydra_main()
    else:
        launch_agent(args)


if __name__ == "__main__":
    app()
