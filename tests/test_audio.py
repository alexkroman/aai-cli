"""Tests for audio module."""

import io

import numpy as np
import pytest
import soundfile as sf

from aai_cli.audio import audio_to_wav_bytes, prepare_wav_bytes, resample_audio, to_pcm_s16le


@pytest.fixture()
def sine_wave():
    """Short 440 Hz sine wave at 16 kHz."""
    t = np.linspace(0, 0.1, 1600, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t), 16000


def test_audio_to_wav_roundtrips(sine_wave):
    audio, sr = sine_wave
    wav_bytes = audio_to_wav_bytes(audio, sr)

    data, rate = sf.read(io.BytesIO(wav_bytes))
    assert rate == sr
    assert len(data) == len(audio)


def test_duck_typed_tensor_converts(sine_wave):
    audio, sr = sine_wave

    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr
            self.ndim = arr.ndim

        def numpy(self):
            return self._arr

        def squeeze(self):
            return self._arr.squeeze()

    assert isinstance(audio_to_wav_bytes(FakeTensor(audio), sr), bytes)


def test_squeezes_multidim(sine_wave):
    audio, sr = sine_wave
    data, _ = sf.read(io.BytesIO(audio_to_wav_bytes(audio.reshape(1, -1), sr)))
    assert data.ndim == 1


def test_prepare_dict_with_array(sine_wave):
    audio, sr = sine_wave
    assert isinstance(prepare_wav_bytes({"array": audio, "sampling_rate": sr}), bytes)


def test_prepare_dict_with_bytes(sine_wave):
    audio, sr = sine_wave
    raw = audio_to_wav_bytes(audio, sr)
    assert prepare_wav_bytes({"bytes": raw}) == raw


def test_prepare_dict_with_path(sine_wave, tmp_path):
    audio, sr = sine_wave
    wav_file = tmp_path / "test.wav"
    sf.write(str(wav_file), audio, sr)
    assert isinstance(prepare_wav_bytes({"path": str(wav_file)}), bytes)


def test_prepare_object_with_attributes(sine_wave):
    audio, sr = sine_wave
    obj = type("Audio", (), {"array": audio, "sampling_rate": sr})()
    assert isinstance(prepare_wav_bytes(obj), bytes)


def test_prepare_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported audio format"):
        prepare_wav_bytes("not audio")


# ---------------------------------------------------------------------------
# resample_audio + to_pcm_s16le
# ---------------------------------------------------------------------------


def test_resample_same_rate():
    data = np.ones(100, dtype=np.float32)
    result = resample_audio(data, 16000, 16000)
    assert np.array_equal(result, data)


def test_resample_downsample():
    data = np.ones(16000, dtype=np.float32)
    result = resample_audio(data, 16000, 8000)
    assert len(result) == 8000


def test_resample_upsample():
    data = np.ones(8000, dtype=np.float32)
    result = resample_audio(data, 8000, 16000)
    assert len(result) == 16000


def test_to_pcm_s16le_returns_bytes(sine_wave):
    audio, sr = sine_wave
    audio_dict = {"array": audio, "sampling_rate": sr}
    pcm = to_pcm_s16le(audio_dict, target_sr=16000)
    assert isinstance(pcm, bytes)
    assert len(pcm) > 0
