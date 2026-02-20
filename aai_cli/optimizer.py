"""Prompt optimization for ASR using DSPy MIPROv2."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import dspy
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress

from .eval import compute_laser_score, compute_wer
from .transcribe import TranscriptionError, set_api_key, transcribe
from .types import AudioData, AudioSample, OptimizationResult

logging.getLogger("dspy").setLevel(logging.WARNING)


class Transcribe(dspy.Signature):
    """Transcribe the audio sample identified by audio_id."""

    audio_id = dspy.InputField(desc="integer index into the audio store")
    transcription = dspy.OutputField()


class ASRModule(dspy.Module):
    """Wrapper that routes DSPy instruction optimization to AssemblyAI."""

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
        self._last_prompt: str | None = None
        self._console = console

    def forward(self, audio_id: int) -> dspy.Prediction:
        prompt = self.predict.signature.instructions
        if prompt != self._last_prompt:
            self._last_prompt = prompt
            if self._console:
                self._console.print(f"[dim]Prompt: {prompt}[/dim]")
        aid = int(audio_id)
        audio = self._audio_store[aid]
        try:
            result = transcribe(
                audio, prompt, self.api_key, speech_model=self.speech_model, api_host=self.api_host
            )
            return dspy.Prediction(transcription=result.text)
        except TranscriptionError as e:
            if self._console:
                self._console.print(f"[red]Transcription error: {e}[/red]")
            return dspy.Prediction(transcription="")


def wer_metric(example: object, pred: object, trace: object = None) -> float:
    """DSPy metric: higher is better, so return 1 - WER."""
    ref = getattr(example, "reference", "") or ""
    hyp = getattr(pred, "transcription", "") or ""
    return max(0.0, 1.0 - compute_wer(ref, hyp))


def laser_metric(example: object, pred: object, trace: object = None) -> float:
    """DSPy metric: LASER score (0-1, higher is better)."""
    ref = getattr(example, "reference", "") or ""
    hyp = getattr(pred, "transcription", "") or ""
    result = compute_laser_score(ref, hyp)
    return result.laser_score


def run_optimization(
    eval_samples: list[AudioSample],
    api_key: str,
    starting_prompt: str,
    iterations: int,
    console: Console,
    candidates_per_step: int = 8,
    llm_model: str = "claude-sonnet-4-6",
    num_threads: int = 12,
    laser: bool = False,
    **_kwargs: object,
) -> OptimizationResult:
    """Run MIPROv2 prompt optimization and return results dict."""
    lm = dspy.LM(f"anthropic/{llm_model}")
    dspy.configure(lm=lm)

    metric = laser_metric if laser else wer_metric
    metric_name = "LASER" if laser else "WER"
    num_trials = iterations * candidates_per_step

    console.print(Markdown("## Starting Optimization (DSPy MIPROv2)"))
    console.print(
        Markdown(
            f"*Samples: {len(eval_samples)}, Trials: {num_trials}, LLM: {llm_model}, Metric: {metric_name}*"
        )
    )

    # Populate audio store and build lightweight trainset.
    audio_store: dict[int, AudioData] = {}
    trainset = []
    for i, s in enumerate(eval_samples):
        audio_store[i] = s.audio
        trainset.append(dspy.Example(audio_id=i, reference=s.reference).with_inputs("audio_id"))

    set_api_key(api_key)
    student = ASRModule(api_key, audio_store=audio_store, console=console)
    student.predict.signature = student.predict.signature.with_instructions(starting_prompt)

    num_candidates = max(candidates_per_step, 5)

    optimizer = dspy.MIPROv2(
        metric=metric,
        auto=None,
        num_candidates=num_candidates,
        num_threads=num_threads,
    )

    # MIPROv2 uses 80% of trainset as valset; cap minibatch_size to fit.
    valset_size = min(1000, max(1, int(len(trainset) * 0.80)))
    minibatch_size = min(35, valset_size)

    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_trials=num_trials,
        minibatch_size=minibatch_size,
    )

    best_prompt = optimized.predict.signature.instructions

    # Final evaluation pass (threaded to match MIPROv2 parallelism)
    def _eval_one(ex: dspy.Example) -> float:
        return metric(ex, optimized(audio_id=ex.audio_id))

    scores: list[float] = []
    with Progress(console=console) as progress:
        task = progress.add_task("Running final evaluation...", total=len(trainset))
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {pool.submit(_eval_one, ex): ex for ex in trainset}
            for future in as_completed(futures):
                scores.append(future.result())
                progress.advance(task)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    if laser:
        # LASER: score is 0-1 (higher=better), display as error rate
        best_err = (1.0 - avg_score) * 100.0
        console.print(Markdown(f"**Best LASER: {best_err:.2f}%**"))
    else:
        # WER metric returns 1-WER, so WER = 1 - score
        best_wer = 1.0 - avg_score
        console.print(Markdown(f"**Best WER: {best_wer * 100:.2f}%**"))

    return OptimizationResult(
        best_prompt=best_prompt,
        best_score=avg_score,
        metric=metric_name,
        starting_prompt=starting_prompt,
        model="assemblyai/universal-3-pro",
        total_eval_samples=len(eval_samples),
        optimizer="DSPy-MIPROv2",
        num_trials=num_trials,
        llm_model=llm_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
