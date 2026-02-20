"""Tests for optimizer module — DSPy components."""

import inspect
import io
from unittest.mock import patch

from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from rich.console import Console

from aai_cli.optimizer import ASRModule, gepa_laser_metric
from aai_cli.types import LaserResult, TranscriptionResult


def _mock_example(reference):
    """Create a mock with attribute access like dspy.Example."""

    class Ex:
        pass

    ex = Ex()
    ex.reference = reference
    return ex


def _mock_pred(transcription):
    class Pred:
        pass

    p = Pred()
    p.transcription = transcription
    return p


@patch("aai_cli.optimizer.transcribe")
def test_asr_module_forward(mock_transcribe):
    """ASRModule.forward should look up audio from audio_store and call transcribe."""
    mock_transcribe.return_value = TranscriptionResult(text="hello world", ttfb=None, ttfs=1.0)
    audio_store = {0: {"path": "test.wav"}}

    module = ASRModule(api_key="test-key", audio_store=audio_store)
    result = module(audio_id=0)

    assert result.transcription == "hello world"
    mock_transcribe.assert_called_once_with(
        {"path": "test.wav"},
        "Transcribe the audio sample identified by audio_id.",
        "test-key",
        speech_model="universal-3-pro",
        api_host=None,
    )


@patch("aai_cli.optimizer.transcribe")
def test_asr_module_forward_error_with_console(mock_transcribe):
    """ASRModule.forward should catch TranscriptionError and log to console."""
    from aai_cli.transcribe import TranscriptionError

    mock_transcribe.side_effect = TranscriptionError("API timeout")
    audio_store = {0: {"path": "test.wav"}}
    buf = io.StringIO()
    console = Console(file=buf, no_color=True)

    module = ASRModule(api_key="test-key", audio_store=audio_store, console=console)
    result = module(audio_id=0)

    assert result.transcription == ""
    assert "Transcription error" in buf.getvalue()


@patch("aai_cli.optimizer.transcribe")
def test_asr_module_forward_error_no_console(mock_transcribe):
    """ASRModule.forward without console should still return empty on error."""
    from aai_cli.transcribe import TranscriptionError

    mock_transcribe.side_effect = TranscriptionError("fail")
    audio_store = {0: {"path": "test.wav"}}

    module = ASRModule(api_key="test-key", audio_store=audio_store, console=None)
    result = module(audio_id=0)
    assert result.transcription == ""


@patch("aai_cli.optimizer.transcribe")
def test_asr_module_forward_no_spoken_audio_silent(mock_transcribe):
    """ASRModule should suppress 'no spoken audio' errors."""
    from aai_cli.transcribe import TranscriptionError

    mock_transcribe.side_effect = TranscriptionError(
        "language_detection cannot be performed on files with no spoken audio."
    )
    audio_store = {0: {"path": "test.wav"}}
    buf = io.StringIO()
    console = Console(file=buf, no_color=True)

    module = ASRModule(api_key="test-key", audio_store=audio_store, console=console)
    result = module(audio_id=0)

    assert result.transcription == ""
    assert "Transcription error" not in buf.getvalue()


@patch("aai_cli.optimizer.transcribe")
def test_asr_module_logs_prompt_change(mock_transcribe):
    """ASRModule should log when the prompt changes."""
    mock_transcribe.return_value = TranscriptionResult(text="ok", ttfb=None, ttfs=1.0)
    audio_store = {0: {"path": "test.wav"}}
    buf = io.StringIO()
    console = Console(file=buf, no_color=True)

    ASRModule._printed_prompts.clear()
    module = ASRModule(api_key="test-key", audio_store=audio_store, console=console)
    module(audio_id=0)

    assert "Prompt:" in buf.getvalue()


# ---------------------------------------------------------------------------
# GEPA LASER metric tests
# ---------------------------------------------------------------------------


@patch("aai_cli.optimizer.compute_laser_score")
def test_gepa_laser_metric_returns_score_with_feedback(mock_laser):
    mock_laser.return_value = LaserResult(
        laser_score=0.7,
        word_count=10,
        no_penalty_errors=["healthcare vs health care"],
        major_errors=["dispatching vs packaging"],
        minor_errors=["recieve vs receive"],
        total_penalty=1.5,
    )
    result = gepa_laser_metric(_mock_example("ref text"), _mock_pred("hyp text"))
    assert isinstance(result, ScoreWithFeedback)
    assert result.score == 0.7
    assert "dispatching vs packaging" in result.feedback
    assert "CRITICAL" in result.feedback
    assert "recieve vs receive" in result.feedback
    assert "MINOR" in result.feedback
    assert "healthcare vs health care" in result.feedback
    assert "ACCEPTABLE" in result.feedback
    assert "1.5/10" in result.feedback


@patch("aai_cli.optimizer.compute_laser_score")
def test_gepa_laser_metric_perfect_score(mock_laser):
    mock_laser.return_value = LaserResult(
        laser_score=1.0,
        word_count=5,
        no_penalty_errors=[],
        major_errors=[],
        minor_errors=[],
        total_penalty=0.0,
    )
    result = gepa_laser_metric(_mock_example("hello world"), _mock_pred("hello world"))
    assert result.score == 1.0
    assert "PERFECT" in result.feedback


def test_gepa_laser_metric_signature():
    """GEPA validates the 5-arg signature; ensure it binds correctly."""
    sig = inspect.signature(gepa_laser_metric)
    # Should bind with all 5 positional/keyword args without error
    sig.bind(
        example=object(),
        pred=object(),
        trace=None,
        pred_name="predict",
        pred_trace=None,
    )


@patch("aai_cli.optimizer.compute_laser_score")
def test_gepa_laser_metric_predictor_level(mock_laser):
    """Metric should work when pred_name is set (predictor-level feedback)."""
    mock_laser.return_value = LaserResult(
        laser_score=0.9,
        word_count=8,
        no_penalty_errors=[],
        major_errors=["word missing"],
        minor_errors=[],
        total_penalty=1.0,
    )
    result = gepa_laser_metric(_mock_example("ref"), _mock_pred("hyp"), pred_name="predict")
    assert isinstance(result, ScoreWithFeedback)
    assert result.score == 0.9
    assert "word missing" in result.feedback
