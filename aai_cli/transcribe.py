"""AssemblyAI transcription helpers (batch + streaming dispatcher)."""

import io
import time

import assemblyai as aai

from .audio import prepare_wav_bytes
from .types import AudioData, TranscriptionResult

SPEECH_MODEL = "universal-3-pro"


class TranscriptionError(Exception):
    """Raised when transcription fails."""


def set_api_key(api_key: str) -> None:
    """Set the AssemblyAI API key once (call before spawning threads)."""
    aai.settings.api_key = api_key


def transcribe(
    audio: AudioData,
    prompt: str,
    api_key: str,
    speech_model: str = SPEECH_MODEL,
    api_host: str | None = None,
) -> TranscriptionResult:
    """Dispatch to batch or streaming transcription based on speech_model."""
    from .streaming import is_streaming_model, transcribe_streaming

    if is_streaming_model(speech_model):
        return transcribe_streaming(
            audio, prompt, api_key, speech_model=speech_model, api_host=api_host
        )
    return transcribe_assemblyai(audio, prompt, api_key)


def transcribe_assemblyai(audio: AudioData, prompt: str, api_key: str) -> TranscriptionResult:
    """Transcribe audio using the AssemblyAI batch API (universal-3-pro).

    Returns TranscriptionResult.
    TTFB is None for batch (no streaming first byte).
    """
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(
        speech_models=[SPEECH_MODEL, "universal-2"],
        language_detection=True,
        prompt=prompt,
    )
    transcriber = aai.Transcriber(config=config)
    wav_bytes = prepare_wav_bytes(audio)
    t0 = time.monotonic()
    try:
        transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
        if transcript.error:
            raise TranscriptionError(transcript.error)
        ttfs = time.monotonic() - t0
        return TranscriptionResult(text=transcript.text or "", ttfb=None, ttfs=ttfs)
    except Exception as e:
        if isinstance(e, TranscriptionError):
            raise
        raise TranscriptionError(str(e)) from e
