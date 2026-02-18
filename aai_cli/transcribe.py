"""AssemblyAI transcription helper."""

import io

import assemblyai as aai

from .audio import prepare_wav_bytes

SPEECH_MODEL = "universal-3-pro"


def transcribe_assemblyai(audio, prompt: str, api_key: str) -> str:
    """Transcribe audio using AssemblyAI with the given prompt."""
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(
        speech_models=[SPEECH_MODEL],
        language_detection=True,
        prompt=prompt,
    )
    transcriber = aai.Transcriber(config=config)
    wav_bytes = prepare_wav_bytes(audio)
    try:
        transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
        return transcript.text or ""
    except Exception as e:
        return f"[transcription error: {e}]"
