"""Tests for optimizer module — DSPy components."""

import io
from unittest.mock import patch

from rich.console import Console

from aai_cli.optimizer import ASRModule, laser_metric, wer_metric
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


def test_wer_metric_perfect():
    assert wer_metric(_mock_example("hello world"), _mock_pred("hello world")) == 1.0


def test_wer_metric_wrong():
    score = wer_metric(_mock_example("hello world"), _mock_pred("goodbye earth"))
    assert 0.0 <= score < 1.0


def test_wer_metric_empty_reference():
    assert wer_metric(_mock_example(""), _mock_pred("anything")) == 1.0


@patch("aai_cli.optimizer.compute_laser_score")
def test_laser_metric(mock_laser):
    mock_laser.return_value = LaserResult(
        laser_score=0.85,
        word_count=5,
        no_penalty_errors=[],
        major_errors=[],
        minor_errors=[],
        total_penalty=0.75,
    )
    score = laser_metric(_mock_example("hello world"), _mock_pred("hello world"))
    assert score == 0.85


@patch("aai_cli.optimizer.compute_laser_score")
def test_laser_metric_low(mock_laser):
    mock_laser.return_value = LaserResult(
        laser_score=0.2,
        word_count=5,
        no_penalty_errors=[],
        major_errors=[],
        minor_errors=[],
        total_penalty=4.0,
    )
    score = laser_metric(_mock_example("turn left"), _mock_pred("turn right"))
    assert score == 0.2


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
def test_asr_module_logs_prompt_change(mock_transcribe):
    """ASRModule should log when the prompt changes."""
    mock_transcribe.return_value = TranscriptionResult(text="ok", ttfb=None, ttfs=1.0)
    audio_store = {0: {"path": "test.wav"}}
    buf = io.StringIO()
    console = Console(file=buf, no_color=True)

    module = ASRModule(api_key="test-key", audio_store=audio_store, console=console)
    module(audio_id=0)

    assert "Prompt:" in buf.getvalue()
