"""Tests for transcribe module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aai_cli.transcribe import transcribe_assemblyai

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
    transcript = MagicMock(text="hello world")
    _mock_aai.Transcriber.return_value.transcribe.return_value = transcript

    assert transcribe_assemblyai(FAKE_AUDIO, "Transcribe verbatim", "fake-key") == "hello world"


def test_transcribe_error(_mock_aai):
    _mock_aai.Transcriber.return_value.transcribe.side_effect = RuntimeError("API down")

    result = transcribe_assemblyai(FAKE_AUDIO, "Transcribe verbatim", "fake-key")
    assert "[transcription error:" in result
