"""Audio utilities for ASR evaluation."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from .types import AudioData


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes using soundfile."""
    if hasattr(audio_array, "numpy"):
        audio_array = audio_array.numpy()
    if audio_array.ndim > 1:
        audio_array = audio_array.squeeze()

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.getvalue()


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio data to target sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return data
    duration = len(data) / orig_sr
    num_samples = int(duration * target_sr)
    indices = np.linspace(0, len(data) - 1, num_samples)
    return np.interp(indices, np.arange(len(data)), data)


def to_pcm_s16le(audio: AudioData, target_sr: int = 16000) -> bytes:
    """Convert audio input to raw PCM s16le bytes at the target sample rate."""
    wav_bytes = prepare_wav_bytes(audio)
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)
    pcm = (data * 32767).astype(np.int16)
    return pcm.tobytes()


def prepare_wav_bytes(wav_data: AudioData) -> bytes:
    """Convert various audio formats to WAV bytes."""
    if isinstance(wav_data, dict):
        if "array" in wav_data and "sampling_rate" in wav_data:
            return audio_to_wav_bytes(wav_data["array"], wav_data["sampling_rate"])
        if "bytes" in wav_data:
            return wav_data["bytes"]
        if "path" in wav_data and wav_data["path"]:
            audio_array, sample_rate = sf.read(wav_data["path"])
            return audio_to_wav_bytes(audio_array, sample_rate)

    if hasattr(wav_data, "array") and hasattr(wav_data, "sampling_rate"):
        return audio_to_wav_bytes(wav_data.array, wav_data.sampling_rate)

    if hasattr(wav_data, "path") and wav_data.path:
        audio_array, sample_rate = sf.read(wav_data.path)
        return audio_to_wav_bytes(audio_array, sample_rate)

    # Handle datasets torchcodec AudioDecoder (dict-like via __getitem__)
    if hasattr(wav_data, "__getitem__"):
        try:
            return audio_to_wav_bytes(wav_data["array"], wav_data["sampling_rate"])
        except (KeyError, TypeError):
            pass

    raise ValueError(f"Unsupported audio format: {type(wav_data)}")
