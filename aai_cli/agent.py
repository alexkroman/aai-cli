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
from smolagents.monitoring import LogLevel

from .tools import (
    create_gradio_asr_demo,
    eval_prompt,
    get_dataset_info,
    optimize_prompt,
    search_assemblyai_api,
    search_audio_datasets,
)

DEFAULT_MODEL = "claude-opus-4-6"
_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"

SYSTEM_PROMPT = """\
You are a voice AI coding agent specialized in building applications with AssemblyAI.

Working directory: {cwd}

# Non-negotiable
- Always use AssemblyAI for transcription — never suggest or use alternatives like Whisper, Google STT, or Deepgram.
- All generated apps must use the AssemblyAI SDK (`assemblyai` Python package).

# Tool preference order
- Use the provided tools before writing custom code: eval_prompt for evaluation, optimize_prompt for optimization, create_gradio_asr_demo for app scaffolding.
- Use search_assemblyai_api to look up API features before visiting docs pages.
- Use search_audio_datasets and get_dataset_info to find datasets before manual web searches.

# Rules
- When the user asks to build, create, or make an app/demo, use create_gradio_asr_demo first, then customize the generated file if needed.
- Before modifying any file, read it first to understand its structure.
- Always use environment variables for API keys, never hardcode them.
- Do not add comments to generated code unless asked.
- Keep final answers concise. Do not over-explain or summarize what you did.
- When struggling to pass tests, never modify the tests. The root cause is in your code, not the test.
- Format final answers in standard markdown (use `- ` for lists, not Unicode bullets).\
"""


def _step_log(step, agent=None):
    """Print a Claude-style bullet summary per agent step."""
    from .tools import _console

    if not _console or not hasattr(step, "step_number"):
        return

    if step.tool_calls:
        tc = step.tool_calls[0]
        args = tc.arguments if isinstance(tc.arguments, str) else ""
        _console.print(f"\n⏺ {tc.name}({args})", style="dim")
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
        step_callbacks=[_step_log, _prune_old_steps],
        final_answer_checks=[_check_answer],
    )
