"""Audio utilities for ASR evaluation."""

import io

import numpy as np
import soundfile as sf


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


def prepare_wav_bytes(wav_data) -> bytes:
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
