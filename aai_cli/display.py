"""Display helpers for eval results."""

import numpy as np
from rich.console import Console

from .types import EvalSampleResult

default_console = Console()


def _latency_stats(values: list[float]) -> str:
    """Format avg/P95/min/max for a list of latency values."""
    arr = np.asarray(values)
    return (
        f"avg {arr.mean():.2f}s | P95 {np.percentile(arr, 95):.2f}s"
        f" | Min {arr.min():.2f}s | Max {arr.max():.2f}s"
    )


def _print_sample_result(
    r: EvalSampleResult, index: int, total: int, laser: bool, console: Console
) -> None:
    """Print per-sample WER, LASER, and REF/HYP lines."""
    wer_pct = r.wer * 100.0
    latency_parts = []
    if r.ttfb is not None:
        latency_parts.append(f"TTFB {r.ttfb:.2f}s")
    if r.ttfs is not None:
        latency_parts.append(f"TTFS {r.ttfs:.2f}s")
    latency_str = " | ".join(latency_parts) if latency_parts else ""
    ds_tag = f" ({r.dataset})" if r.dataset else ""

    if laser and r.laser:
        laser_pct = (1.0 - r.laser.laser_score) * 100.0
        console.print(
            f"[dim][{index}/{total}]{ds_tag} WER: {wer_pct:.1f}% | "
            f"LASER: {laser_pct:.1f}% | {latency_str}[/dim]"
        )
    else:
        console.print(f"[dim][{index}/{total}]{ds_tag} WER: {wer_pct:.1f}% | {latency_str}[/dim]")
    console.print(f"  [green]REF:[/green] {r.reference}")
    console.print(f"  [yellow]HYP:[/yellow] {r.text}")
    if laser and r.laser:
        parts = []
        if r.laser.major_errors:
            parts.append(f"[red]Major: {r.laser.major_errors}[/red]")
        if r.laser.minor_errors:
            parts.append(f"[yellow]Minor: {r.laser.minor_errors}[/yellow]")
        if r.laser.no_penalty_errors:
            parts.append(f"[green]No penalty: {r.laser.no_penalty_errors}[/green]")
        if parts:
            console.print(f"  {' | '.join(parts)}")


def _print_eval_summary(
    results: list[EvalSampleResult],
    total_samples: int,
    speech_model: str,
    api_mode: str,
    laser: bool,
    console: Console,
) -> None:
    """Print latency stats, per-dataset breakdown, and overall WER/LASER."""
    wer_values = [r.wer for r in results]
    avg_wer = sum(wer_values) / len(wer_values) * 100.0

    ttfb_values = [r.ttfb for r in results if r.ttfb is not None]
    ttfs_values = [r.ttfs for r in results if r.ttfs is not None]

    console.print()
    if ttfb_values:
        console.print(f"[bold]TTFB:[/bold] {_latency_stats(ttfb_values)}")
    if ttfs_values:
        console.print(f"[bold]TTFS:[/bold] {_latency_stats(ttfs_values)}")

    # Per-dataset breakdown when multiple datasets are used
    datasets_seen = sorted({r.dataset for r in results if r.dataset})
    if len(datasets_seen) > 1:
        for ds_name in datasets_seen:
            ds_results = [r for r in results if r.dataset == ds_name]
            ds_wers = [r.wer for r in ds_results]
            ds_avg = sum(ds_wers) / len(ds_wers) * 100.0
            line = f"[bold]WER ({ds_name}): {ds_avg:.2f}%[/bold] | Samples: {len(ds_wers)}"
            if laser:
                ds_laser = (
                    1.0 - sum(r.laser.laser_score for r in ds_results if r.laser) / len(ds_results)
                ) * 100.0
                line += f" | LASER: {ds_laser:.2f}%"
            console.print(line)

    console.print(
        f"[bold]WER: {avg_wer:.2f}%[/bold] | Model: {speech_model} ({api_mode}) "
        f"| Samples: {total_samples}"
    )
    if laser:
        laser_scores = [r.laser.laser_score for r in results if r.laser is not None]
        if laser_scores:
            avg_laser = (1.0 - sum(laser_scores) / len(laser_scores)) * 100.0
            console.print(f"[bold]LASER: {avg_laser:.2f}%[/bold]")
