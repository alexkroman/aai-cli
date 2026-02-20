"""Tests for transcribe module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aai_cli.transcribe import TranscriptionError, transcribe_assemblyai

FAKE_AUDIO = {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}


@pytest.fixture()
def _mock_aai():
    """Patch AssemblyAI SDK and wav preparation for all transcription tests."""
    with (
        patch("aai_cli.transcribe.aai") as mock_aai,
        patch("aai_cli.transcribe.prepare_wav_bytes", return_value=b"fake-wav"),
    ):
        mock_aai.TranscriptionConfig.return_value = MagicMock()
        yield mock_aai


def test_transcribe_success(_mock_aai):
    transcript = MagicMock(text="hello world", error=None)
    _mock_aai.Transcriber.return_value.transcribe.return_value = transcript

    result = transcribe_assemblyai(FAKE_AUDIO, "Transcribe verbatim", "fake-key")
    assert result.text == "hello world"
    assert result.ttfb is None
    assert isinstance(result.ttfs, float)


def test_transcribe_error(_mock_aai):
    _mock_aai.Transcriber.return_value.transcribe.side_effect = RuntimeError("API down")

    with pytest.raises(TranscriptionError, match="API down"):
        transcribe_assemblyai(FAKE_AUDIO, "Transcribe verbatim", "fake-key")


# ---------------------------------------------------------------------------
# set_api_key
# ---------------------------------------------------------------------------


@patch("aai_cli.transcribe.aai")
def test_set_api_key(mock_aai):
    from aai_cli.transcribe import set_api_key

    set_api_key("my-key")
    assert mock_aai.settings.api_key == "my-key"


# ---------------------------------------------------------------------------
# transcribe dispatcher
# ---------------------------------------------------------------------------


@patch("aai_cli.transcribe.transcribe_assemblyai")
def test_transcribe_dispatches_batch(mock_batch):
    """Default speech_model should use batch API."""
    from aai_cli.transcribe import transcribe
    from aai_cli.types import TranscriptionResult

    mock_batch.return_value = TranscriptionResult(text="batch result", ttfb=None, ttfs=1.0)
    result = transcribe(FAKE_AUDIO, "prompt", "key", speech_model="universal-3-pro")
    assert result.text == "batch result"
    mock_batch.assert_called_once()


@patch("aai_cli.streaming.transcribe_streaming")
def test_transcribe_dispatches_streaming(mock_stream):
    """Streaming model should dispatch to streaming API."""
    from aai_cli.transcribe import transcribe
    from aai_cli.types import TranscriptionResult

    mock_stream.return_value = TranscriptionResult(text="stream result", ttfb=0.1, ttfs=1.0)
    result = transcribe(FAKE_AUDIO, "prompt", "key", speech_model="u3-pro")
    assert result.text == "stream result"
    mock_stream.assert_called_once()
