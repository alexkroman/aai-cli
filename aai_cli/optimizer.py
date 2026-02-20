"""Prompt optimization for ASR using DSPy GEPA with LASER feedback."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress

from .eval import compute_laser_score
from .transcribe import TranscriptionError, set_api_key, transcribe
from .types import AudioData, AudioSample, OptimizationResult

logging.getLogger("dspy").setLevel(logging.WARNING)


class Transcribe(dspy.Signature):
    """Transcribe the audio sample identified by audio_id."""

    audio_id = dspy.InputField(desc="integer index into the audio store")
    transcription = dspy.OutputField()


class ASRModule(dspy.Module):
    """Wrapper that routes DSPy instruction optimization to AssemblyAI."""

    _printed_prompts: set[str] = set()

    def __init__(
        self,
        api_key: str,
        audio_store: dict[int, AudioData],
        speech_model: str = "universal-3-pro",
        api_host: str | None = None,
        console: Console | None = None,
    ):
        super().__init__()
        self.predict = dspy.Predict(Transcribe)
        self.api_key = api_key
        self.speech_model = speech_model
        self.api_host = api_host
        self._audio_store = audio_store
        self._console = console

    def __deepcopy__(self, memo: dict) -> "ASRModule":
        """Shallow-copy non-serializable attrs so GEPA can clone the module."""
        import copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ("_console", "_audio_store"):
                setattr(new, k, v)  # share by reference
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new

    def forward(self, audio_id: int) -> dspy.Prediction:
        prompt = self.predict.signature.instructions
        if prompt not in ASRModule._printed_prompts:
            ASRModule._printed_prompts.add(prompt)
            if self._console:
                self._console.print(f"[dim]Prompt: {prompt}[/dim]")
        aid = int(audio_id)
        audio = self._audio_store[aid]
        try:
            result = transcribe(
                audio, prompt, self.api_key, speech_model=self.speech_model, api_host=self.api_host
            )
            pred = dspy.Prediction(transcription=result.text)
        except TranscriptionError as e:
            if self._console and "no spoken audio" not in str(e):
                self._console.print(f"[red]Transcription error: {e}[/red]")
            pred = dspy.Prediction(transcription="")

        # Record in DSPy trace so GEPA can reflect on this predictor's execution.
        trace = dspy.settings.trace
        if trace is not None:
            trace.append((self.predict, {"audio_id": aid}, pred))

        return pred


def gepa_laser_metric(
    example: object,
    pred: object,
    trace: object = None,
    pred_name: str | None = None,
    pred_trace: object = None,
) -> ScoreWithFeedback:
    """GEPA-compatible metric: LASER score with structured feedback for reflection."""
    ref = getattr(example, "reference", "") or ""
    hyp = getattr(pred, "transcription", "") or ""
    result = compute_laser_score(ref, hyp)

    parts: list[str] = []
    if result.major_errors:
        parts.append(
            f"CRITICAL — {len(result.major_errors)} major error(s) "
            f"(word substitutions, omissions, or additions that change meaning): "
            f"{'; '.join(result.major_errors)}. "
            f"The prompt should add explicit guidance to prevent these types of errors."
        )
    if result.minor_errors:
        parts.append(
            f"MINOR — {len(result.minor_errors)} minor error(s) "
            f"(spelling, grammar, or formatting that preserves meaning): "
            f"{'; '.join(result.minor_errors)}."
        )
    if result.no_penalty_errors:
        parts.append(
            f"ACCEPTABLE — {len(result.no_penalty_errors)} no-penalty difference(s) "
            f"(variant spellings, discourse markers): "
            f"{'; '.join(result.no_penalty_errors)}."
        )
    parts.append(f"Score: {result.laser_score:.3f} ({result.total_penalty}/{result.word_count} word penalties)")

    if not result.major_errors and not result.minor_errors:
        parts.append("PERFECT — No errors detected. Transcription matches reference exactly.")

    feedback = "\n".join(parts)

    return ScoreWithFeedback(score=result.laser_score, feedback=feedback)


def run_optimization(
    eval_samples: list[AudioSample],
    api_key: str,
    starting_prompt: str,
    iterations: int,
    console: Console,
    llm_model: str = "claude-opus-4-6",
    reflection_model: str | None = None,
    num_threads: int = 12,
    auto: str | None = None,
    **_kwargs: object,
) -> OptimizationResult:
    """Run GEPA prompt optimization with LASER feedback and return results."""
    lm = dspy.LM(f"anthropic/{llm_model}")
    dspy.configure(lm=lm)

    ref_model = reflection_model or llm_model
    reflection_lm = dspy.LM(f"anthropic/{ref_model}", temperature=1.0)

    console.print(Markdown("## Starting Optimization (DSPy GEPA + LASER)"))
    console.print(
        Markdown(
            f"*Samples: {len(eval_samples)}, LLM: {llm_model}, "
            f"Reflection: {ref_model}, Auto: {auto or 'off'}*"
        )
    )

    ASRModule._printed_prompts.clear()

    # Split samples into train (70%) and val (30%) for generalization.
    import random as _random

    shuffled = list(eval_samples)
    _random.Random(42).shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * 0.7))
    train_samples = shuffled[:split_idx]
    val_samples = shuffled[split_idx:]

    console.print(
        Markdown(f"*Train: {len(train_samples)} samples, Val: {len(val_samples)} samples*")
    )

    # Populate audio store and build trainset + valset.
    audio_store: dict[int, AudioData] = {}
    trainset = []
    for i, s in enumerate(train_samples):
        audio_store[i] = s.audio
        trainset.append(dspy.Example(audio_id=i, reference=s.reference).with_inputs("audio_id"))

    valset = []
    for i, s in enumerate(val_samples, start=len(train_samples)):
        audio_store[i] = s.audio
        valset.append(dspy.Example(audio_id=i, reference=s.reference).with_inputs("audio_id"))

    set_api_key(api_key)
    student = ASRModule(api_key, audio_store=audio_store, console=console)
    student.predict.signature = student.predict.signature.with_instructions(starting_prompt)

    gepa_kwargs: dict[str, object] = {
        "metric": gepa_laser_metric,
        "reflection_lm": reflection_lm,
        "num_threads": num_threads,
        "skip_perfect_score": True,
        "warn_on_score_mismatch": False,
        "reflection_minibatch_size": min(len(train_samples), 100),
        "track_stats": True,
    }
    if auto in ("light", "medium", "heavy"):
        gepa_kwargs["auto"] = auto
    else:
        gepa_kwargs["max_full_evals"] = iterations

    optimizer = dspy.GEPA(**gepa_kwargs)  # type: ignore[arg-type]

    optimized = optimizer.compile(student=student, trainset=trainset, valset=valset)

    # Extract baseline val score from GEPA results (index 0 = seed candidate).
    baseline_err: float | None = None
    if hasattr(optimized, "detailed_results") and optimized.detailed_results is not None:
        val_scores = getattr(optimized.detailed_results, "val_aggregate_scores", None)
        if val_scores and len(val_scores) > 0:
            baseline_score = val_scores[0]
            baseline_err = (1.0 - baseline_score) * 100.0
            console.print(Markdown(f"**Baseline Val LASER: {baseline_err:.2f}%**"))

    best_prompt = optimized.predict.signature.instructions

    # Final evaluation on held-out val set (threaded)
    def _eval_one(ex: dspy.Example) -> float:
        result = gepa_laser_metric(ex, optimized(audio_id=ex.audio_id))
        return float(result.score)

    scores: list[float] = []
    with Progress(console=console) as progress:
        task = progress.add_task("Running final eval on val set...", total=len(valset))
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {pool.submit(_eval_one, ex): ex for ex in valset}
            for future in as_completed(futures):
                scores.append(future.result())
                progress.advance(task)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    best_err = (1.0 - avg_score) * 100.0
    console.print(Markdown(f"**Final Val LASER: {best_err:.2f}%**"))
    if baseline_err is not None:
        improvement = baseline_err - best_err
        console.print(Markdown(f"**Improvement: {improvement:+.2f}%**"))

    return OptimizationResult(
        best_prompt=best_prompt,
        best_score=avg_score,
        metric="LASER",
        starting_prompt=starting_prompt,
        model="assemblyai/universal-3-pro",
        total_eval_samples=len(eval_samples),
        optimizer="DSPy-GEPA",
        num_trials=iterations,
        llm_model=llm_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
