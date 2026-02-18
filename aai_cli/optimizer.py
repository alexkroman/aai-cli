"""OPRO-style optimization loop for ASR prompts."""

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import jiwer
import litellm
from rich.console import Console
from rich.markdown import Markdown
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer

from .transcribe import transcribe_assemblyai

_normalize = EnglishTextNormalizer()

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
litellm.suppress_debug_info = True

REFLECTION_PROMPT = """\
You are an expert prompt engineer optimizing instructions for an \
audio-only speech-to-text model (AssemblyAI universal-3-pro). The model \
receives ONLY audio — no images, no text, no documents. The prompt you \
write is the ONLY text input the speech model sees before transcribing.

## Optimization Trajectory
Previous prompts, WER scores, and their worst errors (listed worst to best, \
lower WER is better). The last entry is the current best:

{trajectory}

## Worst Errors from Current Best Prompt
Below are the worst transcription errors. "Reference" is ground truth, \
"Hypothesis" is what the model produced:

{error_samples}

## Your Task

Using the error patterns above, GENERATE a NEW prompt that is DIFFERENT from \
all previous prompts in the trajectory and achieves a WER as low as possible:
- Directly addresses the most common failure modes seen in the errors
- Uses imperative instructions: "Transcribe…", "Preserve…", "Include…"
- Does NOT repeat instructions that are already working well
- Does NOT describe audio content or mention images/documents
- Generalizes across diverse spoken audio (earnings calls, meetings, \
conversations, presentations)
- Is concise — avoid redundant or overlapping instructions

{stagnation_warning}\
Reply with ONLY the improved prompt text on its own line, no explanation \
or commentary."""


META_REFLECTION_PROMPT = """\
You are a meta-optimizer improving the instruction template used by an ASR \
prompt optimizer. The optimizer uses a "reflection prompt" to guide an LLM in \
proposing better speech-to-text prompts for AssemblyAI universal-3-pro.

## Current Reflection Prompt Template
```
{current_reflection_prompt}
```

## Recent Optimization Evidence

### Prompts that IMPROVED WER (good — the reflection prompt guided these well):
{improved_prompts}

### Prompts that DID NOT improve WER (bad — the reflection prompt failed here):
{failed_prompts}

## Your Task

Analyze the evidence above:
- What patterns appear in successful prompts that the reflection prompt encouraged?
- What patterns appear in failed prompts that the reflection prompt should have discouraged?
- What instructions in the reflection prompt are ineffective or missing?

Return an improved version of the reflection prompt template that better guides \
the optimizer toward lower WER. You MUST preserve these exact placeholders in \
your output: {{trajectory}}, {{error_samples}}, and {{stagnation_warning}}.

Reply with ONLY the improved reflection prompt template, no explanation."""


STAGNATION_WARNING = """\
IMPORTANT: The optimization has shown NO improvement for {stale_iterations} \
consecutive iterations. Minor edits are clearly insufficient. You MUST make \
SIGNIFICANT structural changes to the prompt — try a completely different \
approach, reorder instructions, remove ineffective clauses, or reframe the \
task from scratch. Do not make small tweaks.

"""


def _llm_complete(prompt: str, llm_model: str = "claude-sonnet-4-5-20250929") -> str:
    """Call the LLM and return the response text."""
    response = litellm.completion(
        model=f"anthropic/{llm_model}",
        max_tokens=16000,
        num_retries=2,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


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
    """Return the N worst samples by WER (excluding perfect, empty, and >= 100%)."""
    errors = [r for r in results if 0 < r["wer"] < 1.0 and r["hypothesis"].strip()]
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


def _format_trajectory(history: list[dict], max_size: int = 20) -> str:
    """Format top-K prompts by WER as ascending-quality trajectory (OPRO-style)."""
    sorted_history = sorted(history, key=lambda h: h["wer"])
    top_k = sorted_history[:max_size]
    # Show worst-to-best so the LLM sees quality ascending (recency bias)
    top_k.reverse()
    lines = []
    for h in top_k:
        entry = f'WER: {h["wer"]:.2%} — "{h["prompt"]}"'
        for s in h.get("worst_samples", []):
            entry += f'\n  [{s["wer"]:.2%}] ref: "{s["reference"]}" → hyp: "{s["hypothesis"]}"'
        lines.append(entry)
    return "\n\n".join(lines)


def _propose_prompt(
    history: list[dict],
    trajectory_size: int = 20,
    error_samples: str = "",
    llm_model: str = "claude-sonnet-4-5-20250929",
    stagnation_warning: str = "",
    reflection_prompt: str | None = None,
) -> tuple[str, str]:
    """Return (candidate_prompt, full_reflection_prompt) tuple."""
    trajectory = _format_trajectory(history, max_size=trajectory_size)

    template = reflection_prompt if reflection_prompt is not None else REFLECTION_PROMPT
    full_prompt = template.format(
        trajectory=trajectory,
        error_samples=error_samples,
        stagnation_warning=stagnation_warning,
    )

    return _llm_complete(full_prompt, llm_model), full_prompt


def _generate_candidates(
    history: list[dict],
    n_candidates: int,
    trajectory_size: int,
    worst_pool: list[dict],
    llm_model: str = "claude-sonnet-4-5-20250929",
    error_sample_size: int = 10,
    stagnation_warning: str = "",
    reflection_prompt: str | None = None,
) -> list[tuple[str, str]]:
    """Generate N candidate prompts in parallel, each seeing a different random subset of errors."""
    trajectory = _format_trajectory(history, max_size=trajectory_size)
    template = reflection_prompt if reflection_prompt is not None else REFLECTION_PROMPT

    full_prompts: list[str] = []
    for _ in range(n_candidates):
        subset = random.sample(worst_pool, min(error_sample_size, len(worst_pool)))
        error_text = _format_error_samples(subset)
        full_prompts.append(
            template.format(
                trajectory=trajectory,
                error_samples=error_text,
                stagnation_warning=stagnation_warning,
            )
        )

    responses = litellm.batch_completion(
        model=f"anthropic/{llm_model}",
        messages=[[{"role": "user", "content": p}] for p in full_prompts],
        max_tokens=16000,
    )

    return [
        ((r.choices[0].message.content or "").strip(), full_prompts[i])
        for i, r in enumerate(responses)
    ]


def _refine_reflection_prompt(
    current_reflection_prompt: str,
    history: list[dict],
    llm_model: str = "claude-sonnet-4-5-20250929",
) -> str:
    """Ask the LLM to propose an improved reflection prompt based on recent evidence."""
    # Partition history into improved vs. failed based on whether each entry
    # beat the WER of the entry before it (chronological order)
    improved: list[str] = []
    failed: list[str] = []
    for idx in range(1, len(history)):
        prev_best = min(h["wer"] for h in history[:idx])
        entry = history[idx]
        line = f'WER: {entry["wer"]:.2%} — "{entry["prompt"][:120]}"'
        if entry["wer"] < prev_best:
            improved.append(line)
        else:
            failed.append(line)

    improved_text = "\n".join(improved[-10:]) if improved else "(none)"
    failed_text = "\n".join(failed[-10:]) if failed else "(none)"

    meta_prompt = META_REFLECTION_PROMPT.format(
        current_reflection_prompt=current_reflection_prompt,
        improved_prompts=improved_text,
        failed_prompts=failed_text,
    )

    return _llm_complete(meta_prompt, llm_model)


def _validate_reflection_prompt(
    proposed_prompt: str,
    history: list[dict],
    eval_samples: list[dict],
    api_key: str,
    num_threads: int,
    trajectory_size: int,
    llm_model: str = "claude-sonnet-4-5-20250929",
) -> float:
    """Generate a small batch of candidates with the proposed reflection prompt and return best WER."""
    best_entry = min(history, key=lambda h: h["wer"])
    worst_pool = best_entry.get("worst_samples", [])
    if not worst_pool:
        worst_pool = [{"reference": "n/a", "hypothesis": "n/a", "wer": 0.5}]

    candidates = _generate_candidates(
        history,
        n_candidates=2,
        trajectory_size=trajectory_size,
        worst_pool=worst_pool,
        llm_model=llm_model,
        reflection_prompt=proposed_prompt,
    )

    best_wer = 1.0
    for candidate, _ in candidates:
        results = _evaluate_prompt(
            candidate, eval_samples, api_key, num_threads, desc="Meta-validation"
        )
        wer = _mean_wer(results)
        if wer < best_wer:
            best_wer = wer
    return best_wer


def _save_state(state: dict, path: str | None = None) -> None:
    """Persist optimization state to disk for crash recovery."""
    if path is None:
        return
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def run_optimization(
    eval_samples: list[dict],
    api_key: str,
    starting_prompt: str,
    num_threads: int,
    iterations: int,
    console: Console,
    candidates_per_step: int = 8,
    trajectory_size: int = 20,
    resume_history: list[dict] | None = None,
    seed: int = 42,
    llm_model: str = "claude-sonnet-4-5-20250929",
    meta_every: int = 3,
    output_path: str | None = None,
) -> dict:
    """Run OPRO optimization loop and return results dict."""

    current_reflection_prompt = REFLECTION_PROMPT
    reflection_prompt_history: list[dict] = []

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
            "reflection_prompt": current_reflection_prompt,
            "reflection_prompt_history": reflection_prompt_history,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Fixed seed for eval shuffle so WER is comparable across runs;
    # use a separate RNG to avoid making error sample selection deterministic
    rng = random.Random(seed)
    rng.shuffle(eval_samples)

    console.print(Markdown("## Starting Optimization"))
    console.print(
        Markdown(
            f"*Samples: {len(eval_samples)}, Iterations: {iterations}, "
            f"Candidates/step: {candidates_per_step}*"
        )
    )

    if resume_history:
        history = list(resume_history)
        console.print(Markdown(f"**Resuming from previous run ({len(history)} entries)**"))
    else:
        history = []

    # Evaluate starting prompt if no history or if starting fresh
    if not history or history[0]["prompt"] != starting_prompt:
        console.print(Markdown("**Evaluating starting prompt...**"))
        baseline_results = _evaluate_prompt(
            starting_prompt, eval_samples, api_key, num_threads, desc="Baseline"
        )
        baseline_wer = _mean_wer(baseline_results)
        console.print(Markdown(f"Starting WER: **{baseline_wer * 100:.2f}%**"))
        history.append(
            {
                "prompt": starting_prompt,
                "wer": baseline_wer,
                "worst_samples": _worst_samples(baseline_results, n=5),
            }
        )
        best_results = baseline_results
        _save_state(_build_state(history), path=output_path)
    else:
        # Re-evaluate the best prompt from history on current data
        best_prev = min(history, key=lambda h: h["wer"])
        console.print(
            Markdown(f"**Re-evaluating best prompt (prev WER: {best_prev['wer'] * 100:.2f}%)...**")
        )
        best_results = _evaluate_prompt(
            best_prev["prompt"], eval_samples, api_key, num_threads, desc="Resume baseline"
        )
        resume_wer = _mean_wer(best_results)
        console.print(Markdown(f"Baseline WER: **{resume_wer * 100:.2f}%**"))
        # Update existing entry instead of appending a duplicate
        best_prev["wer"] = resume_wer
        best_prev["worst_samples"] = _worst_samples(best_results, n=5)
        _save_state(_build_state(history), path=output_path)

    stale_iterations = 0
    for i in range(1, iterations + 1):
        # Best prompt so far drives the reflection
        best_so_far = min(history, key=lambda h: h["wer"])
        console.print(
            Markdown(
                f"### Iteration {i}/{iterations} (best so far: {best_so_far['wer'] * 100:.2f}%)"
            )
        )

        # Build stagnation warning if no improvement for 3+ iterations
        stagnation_warning = ""
        if stale_iterations >= 3:
            stagnation_warning = STAGNATION_WARNING.format(stale_iterations=stale_iterations)
            console.print(
                Markdown(
                    f"*Stagnant for {stale_iterations} iterations — requesting bolder changes*"
                )
            )

        # Build pool of worst error samples for random subsampling
        worst_pool = _worst_samples(best_results, n=50)

        # Generate N candidates in parallel, each seeing a different error subset
        console.print(Markdown(f"Generating {candidates_per_step} candidates..."))
        candidates = _generate_candidates(
            history,
            candidates_per_step,
            trajectory_size,
            worst_pool,
            llm_model,
            stagnation_warning=stagnation_warning,
            reflection_prompt=current_reflection_prompt,
        )

        # Evaluate all candidates and add to trajectory
        for j, (candidate, _) in enumerate(candidates, 1):
            console.print(Markdown(f'*Candidate {j}: "{candidate[:80]}..."*'))
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
                    "worst_samples": _worst_samples(candidate_results, n=5),
                }
            )
            console.print(Markdown(f"WER: **{candidate_wer * 100:.2f}%**"))

            # Track results from the best candidate for error samples
            if candidate_wer <= min(h["wer"] for h in history):
                best_results = candidate_results

        # Report iteration summary
        new_best = min(history, key=lambda h: h["wer"])
        if new_best["wer"] < best_so_far["wer"]:
            console.print(
                Markdown(
                    f"**New best: {best_so_far['wer'] * 100:.2f}% → {new_best['wer'] * 100:.2f}%**"
                )
            )
            stale_iterations = 0
        else:
            stale_iterations += 1
            console.print(Markdown("*No new best this iteration*"))

        # Meta-optimization: refine the reflection prompt every meta_every iterations
        if meta_every > 0 and i % meta_every == 0:
            console.print(Markdown(f"**Meta-optimization step (iteration {i})...**"))
            proposed = _refine_reflection_prompt(current_reflection_prompt, history, llm_model)
            console.print(Markdown(f'*Proposed reflection prompt: "{proposed[:100]}..."*'))

            # Validate: generate a small batch with the new prompt and compare
            current_best_wer = min(h["wer"] for h in history)
            console.print(Markdown("Validating proposed reflection prompt..."))
            validation_wer = _validate_reflection_prompt(
                proposed,
                history,
                eval_samples,
                api_key,
                num_threads,
                trajectory_size,
                llm_model,
            )
            console.print(
                Markdown(
                    f"Validation WER: **{validation_wer * 100:.2f}%** "
                    f"(current best: {current_best_wer * 100:.2f}%)"
                )
            )

            if validation_wer < current_best_wer:
                console.print(Markdown("**Adopting new reflection prompt**"))
                reflection_prompt_history.append(
                    {"iteration": i, "prompt": current_reflection_prompt, "action": "replaced"}
                )
                current_reflection_prompt = proposed
            else:
                console.print(Markdown("*Keeping current reflection prompt*"))
                reflection_prompt_history.append(
                    {"iteration": i, "prompt": proposed, "action": "rejected"}
                )

        console.print()

        _save_state(_build_state(history, iteration=i), path=output_path)

    # Best prompt across the entire trajectory
    best = min(history, key=lambda h: h["wer"])
    console.print(Markdown(f"**Best WER: {best['wer'] * 100:.2f}%**"))

    return _build_state(history, iteration=iterations)
