"""AssemblyAI streaming transcription helper (v3 WebSocket API).

Streaming models (e.g. u3-pro) use a real-time WebSocket connection,
unlike the batch API models (e.g. universal-3-pro) in transcribe.py.
"""

import io
import threading
import time

import numpy as np
import soundfile as sf
from assemblyai.streaming.v3 import (
    Encoding,
    SpeechModel,
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)

from .audio import prepare_wav_bytes

# Streaming models use the v3 WebSocket client, not the batch API.
STREAMING_SPEECH_MODELS = frozenset(
    {
        "u3-pro",
        "universal-streaming-english",
        "universal-streaming-multilingual",
    }
)

# Only these models support the `prompt` parameter.
PROMPT_SUPPORTED_MODELS = frozenset({"u3-pro"})

TARGET_SAMPLE_RATE = 16000


def is_streaming_model(speech_model: str) -> bool:
    """Return True if the speech model requires the streaming API."""
    return speech_model in STREAMING_SPEECH_MODELS


def _audio_to_pcm_s16le(audio, target_sr: int = TARGET_SAMPLE_RATE) -> bytes:
    """Convert audio input to raw PCM s16le bytes at the target sample rate."""
    wav_bytes = prepare_wav_bytes(audio)
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        duration = len(data) / sr
        num_samples = int(duration * target_sr)
        indices = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(indices, np.arange(len(data)), data)
    pcm = (data * 32767).astype(np.int16)
    return pcm.tobytes()


def transcribe_streaming(
    audio,
    prompt: str,
    api_key: str,
    speech_model: str = "u3-pro",
    api_host: str | None = None,
) -> dict:
    """Transcribe audio using the AssemblyAI streaming API.

    Uses the v3 WebSocket client. Audio is converted to raw PCM s16le at 16kHz
    and streamed in small chunks to simulate real-time delivery.

    Returns {"text": str, "ttfb": float, "ttfs": float}.
    """
    pcm_bytes = _audio_to_pcm_s16le(audio)

    transcripts: list[str] = []
    error: list[str] = []
    done = threading.Event()
    ttfb_val: list[float] = []  # capture first-turn latency
    ttfs_val: list[float] = []  # capture last end_of_turn time

    opts = StreamingClientOptions(api_key=api_key)
    if api_host:
        opts.api_host = api_host
    client = StreamingClient(opts)

    def on_turn(_client, turn):
        if not ttfb_val:
            ttfb_val.append(time.monotonic())
        if turn.end_of_turn:
            transcripts.append(turn.transcript)
            ttfs_val.append(time.monotonic())

    def on_error(_client, err):
        error.append(str(err))

    def on_termination(_client, _event):
        done.set()

    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Error, on_error)
    client.on(StreamingEvents.Termination, on_termination)

    try:
        params = StreamingParameters(
            sample_rate=TARGET_SAMPLE_RATE,
            encoding=Encoding.pcm_s16le,
            speech_model=SpeechModel(speech_model),
        )
        if speech_model in PROMPT_SUPPORTED_MODELS and prompt:
            params.prompt = prompt
        client.connect(params)

        t0 = time.monotonic()

        # Stream in ~100ms chunks (3200 bytes at 16kHz/16-bit) with 20ms
        # inter-chunk delay (matches stt-benchmark methodology).
        chunk_size = 3200
        for i in range(0, len(pcm_bytes), chunk_size):
            client.stream(pcm_bytes[i : i + chunk_size])
            time.sleep(0.02)

        client.disconnect(terminate=True)
        done.wait(timeout=30)
    except Exception as e:
        return {"text": f"[streaming transcription error: {e}]", "ttfb": None, "ttfs": None}

    if error:
        return {"text": f"[streaming transcription error: {error[0]}]", "ttfb": None, "ttfs": None}

    ttfs = ttfs_val[-1] - t0 if ttfs_val else None
    ttfb = ttfb_val[0] - t0 if ttfb_val else None

    return {"text": " ".join(transcripts), "ttfb": ttfb, "ttfs": ttfs}
