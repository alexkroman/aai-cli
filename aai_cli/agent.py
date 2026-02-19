"""Agent construction, callbacks, and system prompt."""

from pathlib import Path

import yaml
from rich.console import Console
from smolagents import (
    ActionStep,
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    UserInputTool,
    VisitWebpageTool,
)
from smolagents.memory import PlanningStep
from smolagents.monitoring import LogLevel

from .tools import (
    create_gradio_asr_demo,
    create_voice_agent,
    edit_file,
    eval_prompt,
    get_dataset_info,
    optimize_prompt,
    read_file,
    search_assemblyai_api,
    search_audio_datasets,
    write_file,
)

DEFAULT_MODEL = "claude-opus-4-6"
_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"
TOOL_NAMES = [
    "eval_prompt",
    "optimize_prompt",
    "search_audio_datasets",
    "get_dataset_info",
    "create_gradio_asr_demo",
    "create_voice_agent",
    "read_file",
    "write_file",
    "edit_file",
    "search_assemblyai_api",
    "final_answer",
]

SYSTEM_PROMPT = """\
You are a voice AI coding agent specialized in building applications with AssemblyAI.

Working directory: {cwd}

# Non-negotiable
- Always use AssemblyAI for transcription — never suggest or use alternatives like Whisper, Google STT, or Deepgram.
- All generated apps must use the AssemblyAI SDK (`assemblyai` Python package).
- IMPORTANT — batch and streaming models use DIFFERENT APIs and are NOT interchangeable:
  - Batch API (upload audio, get transcript back): "universal-3-pro" (default)
  - Streaming API (real-time WebSocket): "u3-pro", "universal-streaming-english", "universal-streaming-multilingual"
  "u3-pro" is NOT a shorthand for "universal-3-pro". If a user says "universal streaming" they mean one of the streaming models. When ambiguous, ask the user to clarify batch vs streaming.

# Tool usage
- Use the provided tools before writing custom code.
- Use search_assemblyai_api to look up API features before visiting docs pages.
- Use search_audio_datasets and get_dataset_info to find datasets before manual web searches.
- Always use read_file before edit_file or write_file on existing files. Never propose changes to code you haven't read.
- Use edit_file to modify existing files. Never overwrite an entire file with write_file when only part of it needs to change.
- Use UserInputTool only for things the user must answer (requirements, preferences, tradeoffs). Never ask the user what you could find out by reading code or searching docs.

# Planning
- For non-trivial tasks (new apps, multi-file changes, unclear requirements), explore existing files with read_file before writing code.
- When planning, structure your thoughts as: what's done, what files were touched, what failed and why, what's next.

# Code quality
- Do not over-engineer. Only make changes that are directly requested or clearly necessary.
- Do not add features, refactor code, or make improvements beyond what was asked.
- Do not create helpers, utilities, or abstractions for one-time operations. Three similar lines of code is better than a premature abstraction.
- Do not add comments, docstrings, or type annotations to code you didn't change.
- When editing code, delete unused imports, variables, and dead code completely. Do not leave commented-out code or "removed" markers.
- Always use environment variables for API keys, never hardcode them.

# Security
- Do not introduce security vulnerabilities — especially command injection in subprocess calls, XSS in web UIs, and path traversal in file operations. If you notice insecure code, fix it immediately.

# Careful actions
- Before overwriting files or running destructive commands, confirm with the user.
- If you encounter unexpected state (e.g., an existing file with user content), investigate before overwriting.

# Communication
- Be direct and objective. Do not praise the user's ideas or add preamble like "Great question!" Focus on delivering working code and factual answers.
- Never estimate how long a task will take. Focus on what needs to be done.
- Keep final answers concise. Do not over-explain or summarize what you did.
- When struggling to pass tests, never modify the tests. The root cause is in your code, not the test.
- Format final answers in standard markdown (use `- ` for lists, not Unicode bullets).\
"""


def _plan_log(step):
    """Print the plan steps."""
    from .tools import _console

    if not _console or not hasattr(step, "plan") or not step.plan:
        return
    _console.print()
    for line in step.plan.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        _console.print(f"  {stripped}", style="dim")


def _step_log(step, agent=None):
    """Print a Claude-style bullet summary per agent step."""
    from .tools import _console

    if not _console or not hasattr(step, "step_number"):
        return

    if step.observations:
        skip = {"Execution logs:", "Last output from code snippet:", ""}
        lines = [ln for ln in step.observations.strip().splitlines() if ln.strip() not in skip]
        for line in lines[:3]:
            _console.print(f"⎿  {line}", style="dim")
        if len(lines) > 3:
            _console.print(f"⎿  … +{len(lines) - 3} lines", style="dim")


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


def create_agent(anthropic_key: str, console: Console, cwd: str) -> CodeAgent:
    """Build and return a configured CodeAgent."""
    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True
    litellm.modify_params = True
    litellm.num_retries = 2
    litellm.request_timeout = 120

    model = LiteLLMModel(
        model_id=f"anthropic/{DEFAULT_MODEL}",
        api_key=anthropic_key,
        max_tokens=8096,
    )

    tools = [
        eval_prompt,
        optimize_prompt,
        search_audio_datasets,
        get_dataset_info,
        create_gradio_asr_demo,
        create_voice_agent,
        read_file,
        write_file,
        edit_file,
        search_assemblyai_api,
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        UserInputTool(),
    ]

    prompt_templates = yaml.safe_load(_PROMPTS_PATH.read_text())

    return CodeAgent(
        tools=tools,
        model=model,
        prompt_templates=prompt_templates,
        instructions=SYSTEM_PROMPT.format(cwd=cwd),
        additional_authorized_imports=[
            "subprocess",
            "pathlib",
            "json",
            "os",
            "shlex",
            "glob",
            "shutil",
        ],
        executor_kwargs={"additional_functions": {"open": open}, "timeout_seconds": 3600},
        max_steps=20,
        planning_interval=5,
        stream_outputs=False,
        max_print_outputs_length=30000,
        verbosity_level=LogLevel.OFF,
        step_callbacks={
            PlanningStep: [_plan_log],
            ActionStep: [_step_log, _prune_old_steps],
        },
        final_answer_checks=[_check_answer],
    )
