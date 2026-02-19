"""AssemblyAI batch transcription helper."""

import io
import time

import assemblyai as aai

from .audio import prepare_wav_bytes

SPEECH_MODEL = "universal-3-pro"


def transcribe_assemblyai(audio, prompt: str, api_key: str) -> dict:
    """Transcribe audio using the AssemblyAI batch API (universal-3-pro).

    Returns {"text": str, "ttfb": None, "ttfs": float}.
    TTFB is None for batch (no streaming first byte).
    """
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(
        speech_models=[SPEECH_MODEL],
        language_detection=True,
        prompt=prompt,
    )
    transcriber = aai.Transcriber(config=config)
    wav_bytes = prepare_wav_bytes(audio)
    t0 = time.monotonic()
    try:
        transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
        ttfs = time.monotonic() - t0
        return {"text": transcript.text or "", "ttfb": None, "ttfs": ttfs}
    except Exception as e:
        ttfs = time.monotonic() - t0
        return {"text": f"[transcription error: {e}]", "ttfb": None, "ttfs": ttfs}
