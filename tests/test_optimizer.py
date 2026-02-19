"""Tests for optimizer module — DSPy components."""

from unittest.mock import patch

from aai_cli.optimizer import ASRModule, _audio_store, wer_metric


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


def testwer_metric_perfect():
    assert wer_metric(_mock_example("hello world"), _mock_pred("hello world")) == 1.0


def testwer_metric_wrong():
    score = wer_metric(_mock_example("hello world"), _mock_pred("goodbye earth"))
    assert 0.0 <= score < 1.0


def testwer_metric_empty_reference():
    assert wer_metric(_mock_example(""), _mock_pred("anything")) == 1.0


@patch("aai_cli.optimizer.transcribe_assemblyai")
def test_asr_module_forward(mock_transcribe):
    """ASRModule.forward should look up audio from _audio_store and call transcribe."""
    mock_transcribe.return_value = {"text": "hello world", "ttfb": None, "ttfs": 1.0}
    _audio_store[0] = {"path": "test.wav"}

    module = ASRModule(api_key="test-key")
    result = module(audio_id=0)

    assert result.transcription == "hello world"
    mock_transcribe.assert_called_once_with(
        {"path": "test.wav"},
        "Transcribe the audio sample identified by audio_id.",
        "test-key",
    )
    _audio_store.clear()
