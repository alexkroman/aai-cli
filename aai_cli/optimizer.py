"""Prompt optimization for ASR using DSPy MIPROv2."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import dspy
from rich.console import Console
from rich.markdown import Markdown

from .eval import compute_laser_score, compute_wer
from .streaming import is_streaming_model, transcribe_streaming
from .transcribe import transcribe_assemblyai

logging.getLogger("dspy").setLevel(logging.WARNING)

# Side-channel for audio data so DSPy never serializes it to the LLM.
_audio_store: dict[int, dict] = {}
_latency_store: dict[int, float] = {}  # audio_id -> seconds


class Transcribe(dspy.Signature):
    """Transcribe the audio sample identified by audio_id."""

    audio_id = dspy.InputField(desc="integer index into the audio store")
    transcription = dspy.OutputField()


class ASRModule(dspy.Module):
    """Wrapper that routes DSPy instruction optimization to AssemblyAI."""

    def __init__(
        self, api_key: str, speech_model: str = "universal-3-pro", api_host: str | None = None
    ):
        super().__init__()
        self.predict = dspy.Predict(Transcribe)
        self.api_key = api_key
        self.speech_model = speech_model
        self.api_host = api_host

    _last_prompt = None
    _console = None

    def forward(self, audio_id):
        prompt = self.predict.signature.instructions
        if prompt != ASRModule._last_prompt:
            ASRModule._last_prompt = prompt
            if ASRModule._console:
                ASRModule._console.print(f"[dim]Prompt: {prompt}[/dim]")
        aid = int(audio_id)
        audio = _audio_store[aid]
        t0 = time.monotonic()
        if is_streaming_model(self.speech_model):
            result = transcribe_streaming(
                audio, prompt, self.api_key, speech_model=self.speech_model, api_host=self.api_host
            )
        else:
            result = transcribe_assemblyai(audio, prompt, self.api_key)
        _latency_store[aid] = time.monotonic() - t0
        return dspy.Prediction(transcription=result["text"])


def wer_metric(example, pred, trace=None):
    """DSPy metric: higher is better, so return 1 - WER."""
    ref = getattr(example, "reference", "") or ""
    hyp = getattr(pred, "transcription", "") or ""
    return max(0.0, 1.0 - compute_wer(ref, hyp))


def laser_metric(example, pred, trace=None):
    """DSPy metric: LASER score (0-1, higher is better)."""
    ref = getattr(example, "reference", "") or ""
    hyp = getattr(pred, "transcription", "") or ""
    result = compute_laser_score(ref, hyp)
    return result["laser_score"]


def run_optimization(
    eval_samples: list[dict],
    api_key: str,
    starting_prompt: str,
    iterations: int,
    console: Console,
    candidates_per_step: int = 8,
    llm_model: str = "claude-sonnet-4-6",
    num_threads: int = 12,
    output_path: str | None = None,
    laser: bool = False,
    **_kwargs,
) -> dict:
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

    # Populate audio store and build lightweight trainset
    _audio_store.clear()
    trainset = []
    for i, s in enumerate(eval_samples):
        _audio_store[i] = s["audio"]
        trainset.append(dspy.Example(audio_id=i, reference=s["reference"]).with_inputs("audio_id"))

    ASRModule._console = console
    ASRModule._last_prompt = None
    student = ASRModule(api_key)
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

    # Final evaluation pass
    console.print(Markdown("**Running final evaluation...**"))
    scores = [metric(ex, optimized(audio_id=ex.audio_id)) for ex in trainset]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    if laser:
        # LASER: score is 0-1 (higher=better), display as error rate
        best_err = (1.0 - avg_score) * 100.0
        console.print(Markdown(f"**Best LASER: {best_err:.2f}%**"))
    else:
        # WER metric returns 1-WER, so WER = 1 - score
        best_wer = 1.0 - avg_score
        console.print(Markdown(f"**Best WER: {best_wer * 100:.2f}%**"))

    result = {
        "best_prompt": best_prompt,
        "best_score": avg_score,
        "metric": metric_name,
        "starting_prompt": starting_prompt,
        "model": "assemblyai/universal-3-pro",
        "total_eval_samples": len(eval_samples),
        "optimizer": "DSPy-MIPROv2",
        "num_trials": num_trials,
        "llm_model": llm_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))

    return result
