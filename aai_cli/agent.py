"""Built-in coding agent using smolagents CodeAgent."""

import os
import subprocess
import sys
import warnings
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from smolagents import ActionStep, CodeAgent, LiteLLMModel, VisitWebpageTool, tool
from smolagents.monitoring import LogLevel

DEFAULT_MODEL = "claude-opus-4-6"

# Module-level state shared by tools and agent
_console: Console | None = None
_cwd: str = "."


def _step_log(step, agent=None):
    """Print a single concise line per agent step."""
    if not _console or not hasattr(step, "step_number"):
        return
    n = step.step_number
    if hasattr(step, "tool_calls") and step.tool_calls:
        names = ", ".join(tc.name for tc in step.tool_calls if hasattr(tc, "name"))
        _console.print(f"  Step {n}: {names}", style="bold")
    else:
        _console.print(f"  Step {n}: planning", style="dim")


def _prune_old_steps(step, agent=None):
    """Prune observations from old steps to keep context from growing unbounded."""
    if not agent or not hasattr(step, "step_number"):
        return
    for prev in agent.memory.steps:
        if (
            isinstance(prev, ActionStep)
            and hasattr(prev, "step_number")
            and prev.step_number < step.step_number - 5
        ):
            if prev.observations and len(prev.observations) > 200:
                prev.observations = prev.observations[:200] + "\n[pruned]"
            if hasattr(prev, "observations_images") and prev.observations_images:
                prev.observations_images = None


def _check_answer(answer, memory=None, agent=None):
    """Validate the final answer is non-empty."""
    return bool(answer and str(answer).strip())


SYSTEM_PROMPT = """\
You are a voice AI coding agent specialized in building applications with AssemblyAI. \
You help users build speech-to-text, audio intelligence, and real-time transcription \
applications.

Working directory: {cwd}

# Tools & capabilities
You can write and execute Python code directly to interact with the filesystem, \
run shell commands, and process data. Use pathlib for file operations, subprocess \
for shell commands, and json for data processing.

You also have these specialized tools:
- eval_prompt: Evaluate a transcription prompt — reports WER against speech datasets.
- optimize_prompt: Run OPRO optimization to find a better transcription prompt.
- search_assemblyai_api: Search the OpenAPI spec for endpoints, parameters, schemas.
- visit_webpage: Fetch any web page as markdown. Use with AssemblyAI docs URLs.

When the user asks about AssemblyAI features or APIs, USE visit_webpage to look up \
current information from https://www.assemblyai.com/docs/ rather than relying on \
memory alone. This ensures accurate, up-to-date answers with correct parameter \
names and code examples.

# Coding practices
- Read before you edit. Understand existing code before suggesting modifications.
- Keep changes minimal and focused. Don't refactor code you weren't asked to change.
- Don't over-engineer. No extra abstractions, helpers, or error handling beyond what's needed.
- Don't add comments, docstrings, or type annotations to code you didn't change.
- Avoid introducing security vulnerabilities (command injection, exposed API keys, etc.).
- Prefer editing existing files over creating new ones.

# Default technology choices
- **Agents**: When asked to build an agent, default to using `smolagents` (the library already \
used by this CLI). Use `ToolCallingAgent` with `LiteLLMModel`.
- **TTS**: When asked to do anything with text-to-speech, default to using **Rime** \
(rime.ai) as the TTS provider.
- **Voice agents**: When asked to build a voice agent, default to using **Pipecat** \
(pipecat.ai) as the framework.

# Prompt eval & optimization
This CLI has built-in tools for evaluating and optimizing AssemblyAI transcription prompts.

Use `eval_prompt` when a user asks to evaluate, test, benchmark, or measure transcription \
quality/WER. Use `optimize_prompt` when they want to improve, optimize, or tune a prompt.

Available datasets (pass as the `dataset` argument):
- "earnings22" — earnings call transcripts (sanchit-gandhi/earnings22_robust_split)
- "peoples" — conversational speech (fixie-ai/peoples_speech, clean split)
- "ami" — meeting recordings (edinburghcstr/ami, IHM microphones)
- "all" — all three (default)

Both tools require `ASSEMBLYAI_API_KEY` and `HF_TOKEN` environment variables.

If the user asks you to WRITE optimization or evaluation code (not use the built-in tools), \
first read the source files as reference templates:
- `aai_cli/optimizer.py` — OPRO optimization loop, LLM-guided candidate generation, \
meta-optimization, WER evaluation, error analysis
- `aai_cli/transcribe.py` — AssemblyAI transcription helper

These show the project's patterns for LLM calls (via litellm), parallel evaluation, \
WER computation, and error formatting. Use them as a starting point.

# AssemblyAI domain knowledge
You are an expert in the AssemblyAI platform and its Python SDK (`assemblyai`).

Key APIs and capabilities:
- **Transcription**: `aai.Transcriber().transcribe(audio_url)` — core speech-to-text
- **Real-time streaming**: `aai.RealtimeTranscriber` — live audio via WebSocket
- **Audio intelligence**: Summarization, sentiment analysis, entity detection, topic detection, \
content moderation, PII redaction, auto chapters, speaker diarization
- **Universal-3 Pro**: Latest model with multilingual support via `speech_model=aai.SpeechModel.best`
- **LLM Gateway**: OpenAI-compatible proxy at `https://llm-gateway.assemblyai.com/v1` — \
use with the OpenAI SDK by setting `base_url` and passing your AssemblyAI API key. \
Model IDs use bare names, e.g. `claude-opus-4-6`, `claude-sonnet-4-5-20250929`.

Common patterns:
- Set API key: `aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]`
- Transcription config: `aai.TranscriptionConfig(speaker_labels=True, language_code="en")`
- Audio sources: local files, URLs, or streams
- Webhook callbacks for async processing
- LLM Gateway usage: `OpenAI(base_url="https://llm-gateway.assemblyai.com/v1", api_key=key)`

When the user asks about voice AI tasks, guide them toward AssemblyAI best practices. \
When writing integration code, always use environment variables for API keys, never hardcode them.

# AssemblyAI docs index
Use visit_webpage with https://www.assemblyai.com/docs/{{path}} to fetch detailed guides:

## Getting Started
- getting-started/transcribe-an-audio-file
- getting-started/transcribe-streaming-audio
- getting-started/universal-3-pro
- getting-started/models

## Pre-Recorded Audio
- pre-recorded-audio/select-the-speech-model
- pre-recorded-audio/prompting
- pre-recorded-audio/keyterms-prompting
- pre-recorded-audio/speaker-diarization
- pre-recorded-audio/multichannel
- pre-recorded-audio/code-switching
- pre-recorded-audio/language-detection
- pre-recorded-audio/custom-spelling
- pre-recorded-audio/filler-words
- pre-recorded-audio/transcript-export-options

## Streaming Audio
- universal-streaming (overview)
- universal-streaming/multichannel-streams
- universal-streaming/multilingual-transcription
- universal-streaming/turn-detection
- universal-streaming/keyterms-prompting

## Voice Agents
- universal-streaming/voice-agents
- universal-streaming/voice-agents/livekit
- universal-streaming/voice-agents/pipecat
- universal-streaming/voice-agents/vapi

## Speech Understanding
- speech-understanding/speaker-identification
- speech-understanding/translation
- speech-understanding/entity-detection
- speech-understanding/sentiment-analysis
- speech-understanding/auto-chapters
- speech-understanding/key-phrases
- speech-understanding/topic-detection
- speech-understanding/summarization

## Content Moderation & PII
- pii-redaction
- content-moderation
- profanity-filtering

## LLM Gateway
- llm-gateway/overview
- llm-gateway/apply-llms-to-audio-files
- llm-gateway/chat-completions
- llm-gateway/conversations
- llm-gateway/tool-calling
- llm-gateway/structured-outputs

## Deployment
- deployment/webhooks
- deployment/account-management

## API Reference
- api-reference/overview
- api-reference/files/upload
- api-reference/transcripts/submit
- api-reference/transcripts/get
- api-reference/transcripts/list
- api-reference/llm-gateway/create-chat-completion
- api-reference/streaming-api/streaming-api\
"""


_AVAILABLE_DATASETS = ["earnings22", "peoples", "ami"]


def _run_aai_cmd(cmd: str, timeout: int = 600) -> str:
    """Run an `aai` CLI command, stream output live, and return it."""
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=_cwd,
    )
    lines: list[str] = []
    try:
        for line in proc.stdout or []:
            line = line.rstrip("\n")
            lines.append(line)
            if _console:
                _console.print(f"    {line}", style="dim")
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        lines.append(f"[timed out after {timeout}s]")

    output = "\n".join(lines)
    if proc.returncode and proc.returncode != 0:
        output += f"\n[exit code: {proc.returncode}]"
    return output or "(no output)"


@tool
def eval_prompt(
    prompt: str,
    max_samples: int = 10,
    num_threads: int = 12,
    dataset: str = "all",
) -> str:
    """Evaluate an AssemblyAI transcription prompt and measure Word Error Rate (WER).

    Sends audio samples to AssemblyAI for transcription using the given prompt,
    then compares the output against ground-truth references. Reports per-sample
    results and overall WER. Lower WER = better transcription quality.

    Use this when a user asks to evaluate, test, or benchmark AssemblyAI transcription,
    measure WER, or compare prompt performance.

    Args:
        prompt: The transcription prompt to evaluate (e.g. "Transcribe verbatim.").
        max_samples: Number of audio samples to transcribe (default: 10 for quick test, use 50+ for reliable results).
        num_threads: Parallel transcription threads (default: 12).
        dataset: Which dataset to evaluate on. One of "earnings22" (earnings calls), "peoples" (conversational), "ami" (meetings), or "all" (all three, default).
    """
    import shlex

    parts = [
        "aai eval",
        f"eval.prompt={shlex.quote(prompt)}",
        f"eval.max_samples={max_samples}",
        f"eval.num_threads={num_threads}",
    ]

    # Filter to a single dataset by removing the others
    if dataset != "all" and dataset in _AVAILABLE_DATASETS:
        for ds in _AVAILABLE_DATASETS:
            if ds != dataset:
                parts.append(f"~datasets.{ds}")

    return _run_aai_cmd(" ".join(parts), timeout=600)


@tool
def optimize_prompt(
    starting_prompt: str,
    iterations: int = 5,
    samples: int = 50,
    candidates_per_step: int = 1,
    num_threads: int = 12,
    trajectory_size: int = 3,
    seed: int = 42,
    llm_model: str = "claude-opus-4-6",
    meta_every: int = 3,
    dataset: str = "all",
) -> str:
    """Optimize an AssemblyAI transcription prompt using OPRO (LLM-guided optimization).

    Iteratively proposes improved transcription prompts by analyzing transcription
    errors and using an LLM to suggest fixes. Each iteration: generates candidate
    prompts, evaluates them against audio datasets, and keeps the best.

    This is long-running — use fewer iterations/samples for quick experiments.

    Use this when a user asks to optimize, improve, or tune a transcription prompt.

    Args:
        starting_prompt: The seed prompt to begin optimizing from.
        iterations: Number of optimization rounds (default: 5).
        samples: Total audio samples to evaluate per candidate (default: 50, split across datasets).
        candidates_per_step: Number of candidate prompts generated per iteration (default: 1).
        num_threads: Parallel transcription threads (default: 12).
        trajectory_size: Number of top prompts shown to the LLM as context (default: 3).
        seed: Random seed for reproducible eval ordering (default: 42).
        llm_model: Model used to generate candidate prompts (default: "claude-opus-4-6"). Options include "claude-sonnet-4-5-20250929".
        meta_every: Run meta-optimization of the reflection prompt every N iterations (default: 3, 0 to disable).
        dataset: Which dataset to use. One of "earnings22" (earnings calls), "peoples" (conversational), "ami" (meetings), or "all" (all three, default).
    """
    import shlex

    parts = [
        "aai optimize",
        f"optimization.starting_prompt={shlex.quote(starting_prompt)}",
        f"optimization.iterations={iterations}",
        f"optimization.samples={samples}",
        f"optimization.candidates_per_step={candidates_per_step}",
        f"optimization.num_threads={num_threads}",
        f"optimization.trajectory_size={trajectory_size}",
        f"optimization.seed={seed}",
        f"optimization.llm_model={shlex.quote(llm_model)}",
        f"optimization.meta_every={meta_every}",
    ]

    if dataset != "all" and dataset in _AVAILABLE_DATASETS:
        for ds in _AVAILABLE_DATASETS:
            if ds != dataset:
                parts.append(f"~datasets.{ds}")

    return _run_aai_cmd(" ".join(parts), timeout=3600)


_OPENAPI_SPEC_PATH = Path(__file__).parent / "data" / "openapi.yml"


@tool
def search_assemblyai_api(query: str) -> str:
    """Search the AssemblyAI OpenAPI spec for endpoints, parameters, and schemas.

    Use this when the user asks about specific API endpoints, request/response
    formats, parameters, or data models. Returns matching sections from the spec.

    Use simple single-word queries for best results (e.g. "transcribe", "speaker_labels").
    Multi-word queries match lines containing ANY of the words, ranked by relevance.

    Args:
        query: Search term (e.g. "transcribe", "speaker_labels", "streaming", "redact_pii").
    """
    if not _OPENAPI_SPEC_PATH.exists():
        return "Error: OpenAPI spec not found at " + str(_OPENAPI_SPEC_PATH)

    spec = _OPENAPI_SPEC_PATH.read_text()
    lines = spec.splitlines()
    words = query.lower().split()
    if not words:
        return "No query provided. Try a keyword like 'transcribe' or 'speaker_labels'."

    # Score each line by how many query words it contains
    scored: list[tuple[int, int]] = []  # (score, line_index)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = sum(1 for w in words if w in line_lower)
        if score > 0:
            scored.append((score, i))

    # Sort by score descending, then by position ascending
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Deduplicate overlapping context windows and collect top matches
    seen_ranges: set[int] = set()
    matches: list[str] = []
    for _score, i in scored:
        if i in seen_ranges:
            continue
        start = max(0, i - 2)
        end = min(len(lines), i + 10)
        seen_ranges.update(range(start, end))
        matches.append("\n".join(lines[start:end]))
        if len(matches) >= 15:
            break

    if not matches:
        return f"No matches for '{query}' in the OpenAPI spec."
    return f"Found {len(matches)} match(es) for '{query}':\n\n" + "\n\n---\n\n".join(matches)


def run_agent(extra_args: list[str] | None = None) -> None:
    """Run the interactive coding agent loop."""
    global _console, _cwd
    _cwd = str(Path.cwd())
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        console = Console()
        console.print("  Missing env var: ANTHROPIC_API_KEY", style="bold red")
        sys.exit(1)

    console = Console()
    _console = console

    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True

    model = LiteLLMModel(
        model_id=f"anthropic/{DEFAULT_MODEL}",
        api_key=anthropic_key,
    )

    tools = [eval_prompt, optimize_prompt, search_assemblyai_api, VisitWebpageTool()]

    agent = CodeAgent(
        tools=tools,
        model=model,
        instructions=SYSTEM_PROMPT.format(cwd=Path.cwd()),
        additional_authorized_imports=[
            "subprocess",
            "pathlib",
            "json",
            "os",
            "shlex",
            "glob",
            "shutil",
        ],
        max_steps=20,
        planning_interval=3,
        stream_outputs=True,
        max_print_outputs_length=30000,
        verbosity_level=LogLevel.OFF,
        step_callbacks=[_step_log, _prune_old_steps],
        final_answer_checks=[_check_answer],
    )

    first_turn = True

    if extra_args:
        first_msg = " ".join(extra_args)
        console.print(f"  > {first_msg}", style="dim")
        try:
            result = agent.run(first_msg)
            console.print()
            console.print(Markdown(str(result)))
            console.print()
        except KeyboardInterrupt:
            console.print("\n  Interrupted.", style="dim")
        except Exception as e:
            console.print(f"  Error: {e}", style="bold red")
        first_turn = False
    else:
        console.print("  Type a message to start. /help for commands.\n", style="dim")

    while True:
        try:
            user_input = console.input("[bold]> [/bold]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n  Goodbye.", style="dim")
            break
        if not user_input.strip():
            continue
        cmd = user_input.strip()
        if cmd == "/clear":
            first_turn = True
            console.print("  Context cleared.", style="dim")
            continue
        if cmd == "/help":
            console.print(
                Markdown(
                    "## Commands\n"
                    "- **/clear** — clear conversation context\n"
                    "- **/help** — show this help\n"
                    "- **Ctrl+C** — interrupt current operation or exit\n"
                    "\n## Agent capabilities\n"
                    "The agent writes and executes Python code directly, including:\n"
                    "- File operations (read, write, edit via pathlib)\n"
                    "- Shell commands (via subprocess)\n"
                    "- **eval_prompt** — evaluate a transcription prompt (WER)\n"
                    "- **optimize_prompt** — OPRO prompt optimization\n"
                    "- **search_assemblyai_api** — search the OpenAPI spec\n"
                    "- **visit_webpage** — fetch any web page as markdown\n"
                )
            )
            continue
        try:
            result = agent.run(user_input, reset=first_turn)
            console.print()
            console.print(Markdown(str(result)))
            console.print()
            first_turn = False
        except KeyboardInterrupt:
            console.print("\n  Interrupted.", style="dim")
        except Exception as e:
            console.print(f"  Error: {e}", style="bold red")
