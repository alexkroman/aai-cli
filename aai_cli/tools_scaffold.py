"""Agent tools for scaffolding Gradio and Pipecat apps."""

from pathlib import Path

from smolagents import tool

from .tools import _resolve_path

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "gradio_app.py.tmpl"
_VOICE_AGENT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "voice_agent.py.tmpl"


def _merge_requirements(req_path: Path, new_deps: list[str]) -> None:
    """Merge *new_deps* into a requirements.txt, preserving existing lines."""
    existing: set[str] = set()
    if req_path.is_file():
        existing = {ln.strip() for ln in req_path.read_text().splitlines() if ln.strip()}
    existing.update(new_deps)
    req_path.write_text("\n".join(sorted(existing)) + "\n")


@tool
def create_gradio_asr_demo(
    title: str = "AssemblyAI Transcription Demo",
    description: str = "Upload audio or record from your microphone to transcribe with AssemblyAI.",
    prompt: str = "",
    output_path: str = "app.py",
) -> str:
    """Build a Gradio web app for AssemblyAI speech-to-text and write it to disk.

    Generates a complete, ready-to-run Python file that launches a Gradio UI for
    transcribing audio via AssemblyAI. Supports microphone recording and file upload.

    **Use this tool first** whenever the user asks to:
    - Build, create, or make a transcription app, demo, or prototype
    - Build a demo for speech-to-text, ASR, or audio transcription
    - Build a voice app, speech-to-text app, or ASR app
    - Create a UI or web interface for audio transcription
    - Make a Gradio demo or prototype for speech recognition
    - Build something with AssemblyAI features (speaker labels, sentiment, etc.)
    - Create a podcast transcriber, meeting transcriber, or call analyzer
    - Demo AssemblyAI capabilities or showcase transcription

    After generating the app, you can further customize the written file if needed.

    Args:
        title: Title shown at the top of the Gradio app.
        description: Subtitle / description shown below the title.
        prompt: AssemblyAI transcription prompt (e.g. "Transcribe verbatim."). Leave empty for default.
        output_path: File path to write the app to (default: "app.py").
    """
    prompt_default = prompt if prompt else ""
    template = _TEMPLATE_PATH.read_text()
    app_code = template.format(
        prompt_default=prompt_default,
        title=title,
        description=description,
    )

    try:
        out = _resolve_path(output_path)
    except ValueError as e:
        return f"Error: {e}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(app_code)

    req_path = out.parent / "requirements.txt"
    _merge_requirements(req_path, ["assemblyai>=0.30", "gradio>=5.0"])

    return (
        f"Wrote Gradio ASR demo to {out}\n"
        f"Wrote {req_path}\n\n"
        f"Run it with:\n"
        f"  pip install -r requirements.txt\n"
        f"  ASSEMBLYAI_API_KEY=your-key python {output_path}"
    )


@tool
def create_voice_agent(
    title: str = "AssemblyAI Voice Agent",
    description: str = "Talk to an AI assistant powered by AssemblyAI, Claude, and Rime.",
    system_prompt: str = "You are a helpful voice assistant. Keep responses brief - 1-2 sentences.",
    rime_speaker: str = "cove",
    rime_model: str = "mistv2",
    output_path: str = "voice_agent.py",
) -> str:
    """Build a real-time voice agent app using Pipecat and write it to disk.

    Generates a complete Python file that runs a conversational voice agent pipeline:
    AssemblyAI streaming STT → Claude LLM → Rime TTS, orchestrated by Pipecat.
    Supports WebRTC (local browser) and Twilio transports out of the box.

    Use this tool when the user asks to:
    - Build a voice agent, voice assistant, or conversational AI
    - Create a real-time speech-to-speech app or demo
    - Build something with streaming transcription and TTS
    - Make a talk-to-AI or voice chat application

    After generating the app, you can further customize the written file if needed.

    Args:
        title: Title used in the module docstring.
        description: Description used in the module docstring.
        system_prompt: System prompt for the Claude LLM that controls assistant personality.
        rime_speaker: Rime TTS voice to use (default: "cove").
        rime_model: Rime TTS model to use (default: "mistv2").
        output_path: File path to write the app to (default: "voice_agent.py").
    """
    template = _VOICE_AGENT_TEMPLATE_PATH.read_text()
    app_code = template.format(
        title=title,
        description=description,
        system_prompt=system_prompt,
        rime_speaker=rime_speaker,
        rime_model=rime_model,
    )

    try:
        out = _resolve_path(output_path)
    except ValueError as e:
        return f"Error: {e}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(app_code)

    req_path = out.parent / "requirements.txt"
    _merge_requirements(
        req_path,
        [
            "pipecat-ai[assemblyai,anthropic,rime,silero,webrtc]",
            "pipecat-ai-small-webrtc-prebuilt",
            "python-dotenv",
        ],
    )

    return (
        f"Wrote voice agent to {out}\n"
        f"Wrote {req_path}\n\n"
        f"Run it with:\n"
        f"  pip install -r requirements.txt\n"
        f"  python {output_path} --transport webrtc\n\n"
        f"Set env vars: ASSEMBLYAI_API_KEY, ANTHROPIC_API_KEY, RIME_API_KEY\n"
        f"(or add them to a .env file)"
    )
