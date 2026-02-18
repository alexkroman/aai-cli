"""OPRO-style optimization loop for ASR prompts."""

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import anthropic
import jiwer
from rich.console import Console
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer

from .transcribe import transcribe_assemblyai

_normalize = EnglishTextNormalizer()

GRADIENT_PROMPT = """\
You are a gradient engine for optimizing speech-to-text prompts. \
Given a prompt and a transcription error, explain why the prompt \
caused this specific error and what instruction change would fix it.

## Prompt
{prompt}

## Error (WER: {wer:.2%})
Reference:  {reference}
Hypothesis: {hypothesis}

In 1-2 sentences: Why did this prompt produce this error, and what \
specific instruction would fix it? Be concrete and actionable."""

AGGREGATE_GRADIENT_PROMPT = """\
You are part of an optimization system that improves speech-to-text prompts. \
Below are individual gradient feedback signals from {n} transcription errors. \
Critically aggregate them into a single concise summary.

{gradients}

Summarize the core failure patterns and the most impactful instruction changes \
in 2-4 sentences. Focus on what is most frequent and actionable. \
Do NOT propose a new prompt — only summarize the feedback."""

REFLECTION_PROMPT = """\
You are an expert prompt engineer optimizing instructions for an \
audio-only speech-to-text model (AssemblyAI universal-3-pro). The model \
receives ONLY audio — no images, no text, no documents. The prompt you \
write is the ONLY text input the speech model sees before transcribing.

## Optimization Trajectory
Previous prompts, WER scores, and gradient feedback (listed worst to best, \
lower WER is better). The last entry is the current best:

{trajectory}

## Worst Errors from Current Best Prompt
Below are the worst transcription errors. "Reference" is ground truth, \
"Hypothesis" is what the model produced:

{error_samples}

## Your Task

Using the gradient feedback and error samples above, GENERATE an improved \
prompt that:
- Directly addresses the most common failure modes identified in the gradients
- Uses imperative instructions: "Transcribe…", "Preserve…", "Include…"
- Does NOT repeat instructions that are already working well
- Does NOT describe audio content or mention images/documents
- Generalizes across diverse spoken audio (earnings calls, meetings, \
conversations, presentations)
- Is concise — avoid redundant or overlapping instructions

Reply with ONLY the improved prompt text on its own line, no explanation \
or commentary."""


def _compute_wer(reference: str, hypothesis: str) -> float:
    ref = _normalize(reference or "")
    hyp = _normalize(hypothesis or "")
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    return jiwer.wer(ref, hyp)


def _evaluate_prompt(
    prompt: str,
    samples: list[dict],
    api_key: str,
    num_threads: int,
    desc: str = "Transcribing",
) -> list[dict]:
    """Transcribe all samples with the given prompt and compute per-sample WER."""
    results = [None] * len(samples)

    def _transcribe_one(idx: int, sample: dict):
        hyp = transcribe_assemblyai(sample["audio"], prompt, api_key)
        wer = _compute_wer(sample["reference"], hyp)
        return idx, {"reference": sample["reference"], "hypothesis": hyp, "wer": wer}

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(_transcribe_one, i, s): i for i, s in enumerate(samples)}
        with tqdm(total=len(samples), desc=f"  {desc}", unit="sample") as pbar:
            for future in as_completed(futures):
                idx, res = future.result()
                results[idx] = res
                pbar.update(1)

    return results


def _mean_wer(results: list[dict]) -> float:
    if not results:
        return 1.0
    return sum(r["wer"] for r in results) / len(results)


def _worst_samples(results: list[dict], n: int = 10) -> list[dict]:
    """Return the N worst samples by WER (excluding perfect scores)."""
    errors = [r for r in results if r["wer"] > 0]
    ranked = sorted(errors, key=lambda r: r["wer"], reverse=True)
    return ranked[:n]


def _format_error_samples(samples: list[dict]) -> str:
    """Format worst samples as ref/hyp pairs for the reflection prompt."""
    blocks = []
    for i, r in enumerate(samples, 1):
        blocks.append(
            f"--- Sample {i} (WER {r['wer']:.2%}) ---\n"
            f"Reference:  {r['reference']}\n"
            f"Hypothesis: {r['hypothesis']}"
        )
    return "\n\n".join(blocks)


def _compute_gradient(prompt: str, sample: dict, anthropic_key: str) -> str:
    """Ask Haiku why the prompt caused this specific error and how to fix it."""
    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": GRADIENT_PROMPT.format(
                    prompt=prompt,
                    wer=sample["wer"],
                    reference=sample["reference"],
                    hypothesis=sample["hypothesis"],
                ),
            }
        ],
    )
    return message.content[0].text.strip()


def _aggregate_gradients(gradients: list[str], anthropic_key: str) -> str:
    """Aggregate multiple per-error gradients into one concise summary."""
    numbered = "\n".join(f"{i}. {g}" for i, g in enumerate(gradients, 1))
    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": AGGREGATE_GRADIENT_PROMPT.format(
                    n=len(gradients),
                    gradients=numbered,
                ),
            }
        ],
    )
    return message.content[0].text.strip()


def _compute_aggregated_gradient(
    results: list[dict], prompt: str, anthropic_key: str, n: int = 5
) -> str:
    """Compute per-error gradients in parallel, then aggregate into one summary."""
    errors = [r for r in results if r["wer"] > 0]
    ranked = sorted(errors, key=lambda r: r["wer"], reverse=True)
    worst = ranked[:n]

    if not worst:
        return ""

    # Compute individual gradients in parallel
    gradients: list[str] = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_compute_gradient, prompt, r, anthropic_key) for r in worst]
        for future in as_completed(futures):
            gradients.append(future.result())

    # Aggregate into a single signal
    return _aggregate_gradients(gradients, anthropic_key)


def _format_trajectory(history: list[dict], max_size: int = 20) -> str:
    """Format top-K prompts by WER as ascending-quality trajectory (OPRO-style)."""
    sorted_history = sorted(history, key=lambda h: h["wer"])
    top_k = sorted_history[:max_size]
    # Show worst-to-best so the LLM sees quality ascending (recency bias)
    top_k.reverse()
    lines = []
    for h in top_k:
        entry = f'WER: {h["wer"]:.2%} — "{h["prompt"]}"'
        gradient = h.get("gradient", "")
        if gradient:
            entry += f"\n  Gradient: {gradient}"
        lines.append(entry)
    return "\n\n".join(lines)


def _propose_prompt(
    history: list[dict],
    anthropic_key: str,
    trajectory_size: int = 20,
    error_samples: str = "",
) -> tuple[str, str]:
    """Return (candidate_prompt, full_reflection_prompt) tuple."""
    trajectory = _format_trajectory(history, max_size=trajectory_size)

    full_prompt = REFLECTION_PROMPT.format(
        trajectory=trajectory,
        error_samples=error_samples,
    )

    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        temperature=1.0,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": full_prompt}],
    )
    # Extract the text block (skip thinking blocks)
    for block in message.content:
        if block.type == "text":
            return block.text.strip(), full_prompt
    return "", full_prompt


def _generate_candidates(
    history: list[dict],
    anthropic_key: str,
    n_candidates: int,
    trajectory_size: int,
    error_samples: str = "",
) -> list[tuple[str, str]]:
    """Generate N candidate prompts in parallel. Returns list of (candidate, full_prompt)."""
    results: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=n_candidates) as pool:
        futures = [
            pool.submit(
                _propose_prompt,
                history,
                anthropic_key,
                trajectory_size,
                error_samples,
            )
            for _ in range(n_candidates)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _save_state(state: dict, path: str = "outputs/optimization_state.json") -> None:
    """Persist optimization state to disk for crash recovery."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def run_optimization(
    eval_samples: list[dict],
    api_key: str,
    anthropic_key: str,
    starting_prompt: str,
    num_threads: int,
    iterations: int,
    console: Console,
    candidates_per_step: int = 8,
    trajectory_size: int = 20,
    resume_history: list[dict] | None = None,
    seed: int = 42,
) -> dict:
    """Run OPRO optimization loop and return results dict."""

    def _build_state(history: list[dict], iteration: int = 0) -> dict:
        best = min(history, key=lambda h: h["wer"]) if history else {}
        return {
            "best_prompt": best.get("prompt", ""),
            "best_wer": best.get("wer", 1.0),
            "starting_prompt": starting_prompt,
            "starting_wer": history[0]["wer"] if history else 1.0,
            "model": "assemblyai/universal-3-pro",
            "total_eval_samples": len(eval_samples),
            "optimizer": "OPRO",
            "iterations": iterations,
            "iterations_completed": iteration,
            "candidates_per_step": candidates_per_step,
            "trajectory_size": trajectory_size,
            "seed": seed,
            "history": history,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Fixed seed for eval shuffle so WER is comparable across runs;
    # use a separate RNG to avoid making error sample selection deterministic
    rng = random.Random(seed)
    rng.shuffle(eval_samples)

    console.print("\n  [bold progress]Starting optimization...[/bold progress]")
    console.print(
        f"  Samples: {len(eval_samples)}, Iterations: {iterations}, "
        f"Candidates/step: {candidates_per_step}\n"
    )

    if resume_history:
        history = list(resume_history)
        console.print(f"  [progress]Resuming from previous run ({len(history)} entries)[/progress]")
    else:
        history = []

    # Evaluate starting prompt if no history or if starting fresh
    if not history or history[0]["prompt"] != starting_prompt:
        console.print("  [progress]Evaluating starting prompt...[/progress]")
        baseline_results = _evaluate_prompt(
            starting_prompt, eval_samples, api_key, num_threads, desc="Baseline"
        )
        baseline_wer = _mean_wer(baseline_results)
        console.print(f"  Starting WER: [bold]{baseline_wer * 100:.2f}%[/bold]")
        console.print("  [progress]Computing gradients...[/progress]")
        history.append(
            {
                "prompt": starting_prompt,
                "wer": baseline_wer,
                "gradient": _compute_aggregated_gradient(
                    baseline_results, starting_prompt, anthropic_key
                ),
            }
        )
        best_results = baseline_results
        _save_state(_build_state(history))
    else:
        # Re-evaluate the best prompt from history on current data
        best_prev = min(history, key=lambda h: h["wer"])
        console.print(
            f"  [progress]Re-evaluating best prompt "
            f"(prev WER: {best_prev['wer'] * 100:.2f}%)...[/progress]"
        )
        best_results = _evaluate_prompt(
            best_prev["prompt"], eval_samples, api_key, num_threads, desc="Resume baseline"
        )
        resume_wer = _mean_wer(best_results)
        console.print(f"  Baseline WER: [bold]{resume_wer * 100:.2f}%[/bold]")
        console.print("  [progress]Computing gradients...[/progress]")
        # Update existing entry instead of appending a duplicate
        best_prev["wer"] = resume_wer
        best_prev["gradient"] = _compute_aggregated_gradient(
            best_results, best_prev["prompt"], anthropic_key
        )
        _save_state(_build_state(history))

    for i in range(1, iterations + 1):
        # Best prompt so far drives the reflection
        best_so_far = min(history, key=lambda h: h["wer"])
        console.print(
            f"  [progress]Iteration {i}/{iterations}[/progress] "
            f"(best so far: {best_so_far['wer'] * 100:.2f}%)"
        )

        # Build error samples from current best's evaluation results
        worst = _worst_samples(best_results)
        error_samples_text = _format_error_samples(worst)

        # Generate N candidates in parallel
        console.print(f"    Generating {candidates_per_step} candidates...")
        candidates = _generate_candidates(
            history,
            anthropic_key,
            candidates_per_step,
            trajectory_size,
            error_samples_text,
        )

        # Evaluate all candidates and add to trajectory
        for j, (candidate, _) in enumerate(candidates, 1):
            console.print(f'    [muted]Candidate {j}: "{candidate[:80]}..."[/muted]')
            candidate_results = _evaluate_prompt(
                candidate,
                eval_samples,
                api_key,
                num_threads,
                desc=f"Iter {i}/{iterations} [{j}/{candidates_per_step}]",
            )
            candidate_wer = _mean_wer(candidate_results)
            history.append(
                {
                    "prompt": candidate,
                    "wer": candidate_wer,
                    "gradient": _compute_aggregated_gradient(
                        candidate_results, candidate, anthropic_key
                    ),
                }
            )
            console.print(f"      WER: {candidate_wer * 100:.2f}%")

            # Track results from the best candidate for error samples
            if candidate_wer <= min(h["wer"] for h in history):
                best_results = candidate_results

        # Report iteration summary
        new_best = min(history, key=lambda h: h["wer"])
        if new_best["wer"] < best_so_far["wer"]:
            console.print(
                f"    [success]New best: {best_so_far['wer'] * 100:.2f}% → "
                f"{new_best['wer'] * 100:.2f}%[/success]"
            )
        else:
            console.print("    [warning]No new best this iteration[/warning]")
        console.print()

        _save_state(_build_state(history, iteration=i))

    # Best prompt across the entire trajectory
    best = min(history, key=lambda h: h["wer"])
    console.print(f"  [bold success]Best WER: {best['wer'] * 100:.2f}%[/bold success]\n")

    return _build_state(history, iteration=iterations)
