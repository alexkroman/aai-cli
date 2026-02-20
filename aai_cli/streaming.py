"""AssemblyAI streaming transcription helper (v3 WebSocket API).

Streaming models (e.g. u3-pro) use a real-time WebSocket connection,
unlike the batch API models (e.g. universal-3-pro) in transcribe.py.
"""

import threading
import time

from assemblyai.streaming.v3 import (
    Encoding,
    SpeechModel,
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)

from .audio import to_pcm_s16le
from .transcribe import TranscriptionError
from .types import AudioData, TranscriptionResult

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


def _build_streaming_params(
    speech_model: str, prompt: str, sample_rate: int = TARGET_SAMPLE_RATE
) -> StreamingParameters:
    """Build StreamingParameters, adding prompt only for supported models."""
    params = StreamingParameters(
        sample_rate=sample_rate,
        encoding=Encoding.pcm_s16le,
        speech_model=SpeechModel(speech_model),
    )
    if speech_model in PROMPT_SUPPORTED_MODELS and prompt:
        params.prompt = prompt
    return params


def transcribe_streaming(
    audio: AudioData,
    prompt: str,
    api_key: str,
    speech_model: str = "u3-pro",
    api_host: str | None = None,
) -> TranscriptionResult:
    """Transcribe audio using the AssemblyAI streaming API.

    Uses the v3 WebSocket client. Audio is converted to raw PCM s16le at 16kHz
    and streamed in small chunks to simulate real-time delivery.

    Returns TranscriptionResult.
    """
    pcm_bytes = to_pcm_s16le(audio, TARGET_SAMPLE_RATE)

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
        params = _build_streaming_params(speech_model, prompt)
        client.connect(params)

        t0 = time.monotonic()

        # Stream in ~100ms chunks (3200 bytes at 16kHz/16-bit) with 20ms
        # inter-chunk delay (matches stt-benchmark methodology).
        chunk_size = 3200
        for i in range(0, len(pcm_bytes), chunk_size):
            client.stream(pcm_bytes[i : i + chunk_size])
            time.sleep(0.02)

        client.disconnect(terminate=True)
        if not done.wait(timeout=30):
            raise TranscriptionError("timed out waiting for response")
    except Exception as e:
        if isinstance(e, TranscriptionError):
            raise
        raise TranscriptionError(str(e)) from e

    if error:
        raise TranscriptionError(error[0])

    ttfs = ttfs_val[-1] - t0 if ttfs_val else None
    ttfb = ttfb_val[0] - t0 if ttfb_val else None

    return TranscriptionResult(text=" ".join(transcripts), ttfb=ttfb, ttfs=ttfs)
