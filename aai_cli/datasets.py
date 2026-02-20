"""Dataset loading and configuration helpers."""

import typer
from datasets import Audio, load_dataset
from huggingface_hub import login as hf_login
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from .display import default_console
from .types import AudioSample, DatasetOptions


def _filter_datasets(cfg: DictConfig, dataset: str | None) -> DictConfig:
    """Keep only named dataset, or all if 'all'/None."""
    if not dataset or dataset == "all":
        return cfg
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected config to be a dict, got {type(container).__name__}")
    if dataset not in container.get("datasets", {}):
        raise KeyError(f"Unknown dataset '{dataset}'. Available: {list(container['datasets'])}")
    container["datasets"] = {dataset: container["datasets"][dataset]}
    return OmegaConf.create(container)


def _apply_hf_dataset_override(cfg: DictConfig, opts: DatasetOptions) -> DictConfig:
    """Replace cfg.datasets with a single ad-hoc HF dataset entry."""
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected config to be a dict, got {type(container).__name__}")
    container["datasets"] = {
        "custom": {
            "path": opts.hf_dataset,
            "config": opts.hf_config,
            "audio_column": opts.audio_column,
            "text_column": opts.text_column,
            "split": opts.split,
        }
    }
    return OmegaConf.create(container)


def _resolve_dataset_config(cfg: DictConfig, opts: DatasetOptions) -> DictConfig:
    """Apply HF dataset override or filter to named dataset."""
    if opts.hf_dataset:
        return _apply_hf_dataset_override(cfg, opts)
    return _filter_datasets(cfg, opts.dataset)


def _validate_dataset_args(hf_dataset: str | None, dataset: str | None) -> None:
    """Exit if both --hf-dataset and --dataset are provided."""
    if hf_dataset and dataset:
        default_console.print(
            "--hf-dataset and --dataset are mutually exclusive.", style="bold red"
        )
        raise typer.Exit(code=1)


def load_dataset_samples(
    ds_name: str,
    ds_cfg: DictConfig,
    num_samples: int,
    hf_token: str,
    console: Console | None = None,
) -> list[AudioSample]:
    """Load samples from one HF dataset."""
    console = console or default_console
    console.print(f"Loading [italic]{ds_name}[/italic]...")

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

    samples: list[AudioSample] = []
    for sample in dataset:
        reference = sample[text_col]
        if isinstance(reference, str) and reference.strip() == "ignore_time_segment_in_scoring":
            continue
        if isinstance(reference, str) and "inaudible" in reference.lower():
            continue
        samples.append(AudioSample(audio=sample[audio_col], reference=reference))
        if len(samples) >= num_samples:
            break

    console.print(f"Loaded [bold]{len(samples)}[/bold] samples from [italic]{ds_name}[/italic]")
    return samples


def load_all_datasets(
    cfg: DictConfig,
    hf_token: str,
    total_samples: int | None = None,
    console: Console | None = None,
) -> list[AudioSample]:
    """Split total samples evenly across configured datasets and collect them."""
    console = console or default_console
    hf_login(token=hf_token, add_to_git_credential=False)
    ds_names = list(cfg.datasets.keys())
    total = total_samples if total_samples is not None else cfg.optimization.samples
    per_ds = total // len(ds_names)
    remainder = total % len(ds_names)

    all_samples: list[AudioSample] = []
    for i, name in enumerate(ds_names):
        n = per_ds + (1 if i < remainder else 0)
        samples = load_dataset_samples(name, cfg.datasets[name], n, hf_token, console=console)
        for s in samples:
            s.dataset = name
        all_samples.extend(samples)

    console.print(
        f"Total: [bold]{len(all_samples)}[/bold] samples from [bold]{len(ds_names)}[/bold] datasets"
    )
    return all_samples
