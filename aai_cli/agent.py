"""Agent construction, callbacks, and system prompt."""

import functools
from collections.abc import Callable
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

from .tools_eval import eval_prompt, optimize_prompt
from .tools_filesystem import edit_file, read_file, write_file
from .tools_scaffold import create_gradio_asr_demo, create_voice_agent
from .tools_search import get_dataset_info, search_assemblyai_api, search_audio_datasets

DEFAULT_MODEL = "claude-opus-4-6"
_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"
_SYSTEM_PROMPT_PATH = Path(__file__).parent / "templates" / "system_prompt.txt"


def _make_plan_log(console: Console) -> Callable[[PlanningStep], None]:
    """Create a callback that prints plan steps to the given console."""

    def _plan_log(step: PlanningStep) -> None:
        if not hasattr(step, "plan") or not step.plan:
            return
        console.print()
        for line in step.plan.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            console.print(f"  {stripped}", style="dim")

    return _plan_log


def _make_step_log(
    console: Console,
) -> Callable[[ActionStep, CodeAgent | None], None]:
    """Create a callback that prints step summaries to the given console."""

    def _step_log(step: ActionStep, agent: CodeAgent | None = None) -> None:
        if not hasattr(step, "step_number"):
            return

        if step.observations:
            skip = {"Execution logs:", "Last output from code snippet:", ""}
            lines = [ln for ln in step.observations.strip().splitlines() if ln.strip() not in skip]
            for line in lines[:3]:
                console.print(f"⎿  {line}", style="dim")
            if len(lines) > 3:
                console.print(f"⎿  … +{len(lines) - 3} lines", style="dim")

    return _step_log


def _prune_old_steps(step: ActionStep, agent: CodeAgent | None = None) -> None:
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


def _check_answer(
    answer: str | None, memory: object = None, agent: CodeAgent | None = None
) -> bool:
    """Validate the final answer is non-empty."""
    return bool(answer and str(answer).strip())


@functools.lru_cache(maxsize=1)
def _configure_litellm() -> None:
    """Set litellm globals once."""
    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True
    litellm.modify_params = True
    litellm.num_retries = 2
    litellm.request_timeout = 120


def create_agent(
    anthropic_key: str,
    console: Console,
    cwd: str,
    prompts_path: Path | None = None,
    system_prompt_path: Path | None = None,
) -> tuple[CodeAgent, list[str]]:
    """Build and return a configured CodeAgent and the list of tool names."""
    _configure_litellm()

    prompts_path = prompts_path or _PROMPTS_PATH
    system_prompt_path = system_prompt_path or _SYSTEM_PROMPT_PATH

    model = LiteLLMModel(
        model_id=f"anthropic/{DEFAULT_MODEL}",
        api_key=anthropic_key,
        max_tokens=8192,
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

    tool_names = [getattr(t, "name", t.__class__.__name__) for t in tools] + ["final_answer"]

    prompt_templates = yaml.safe_load(prompts_path.read_text())

    agent = CodeAgent(
        tools=tools,
        model=model,
        prompt_templates=prompt_templates,
        instructions=system_prompt_path.read_text().format(cwd=cwd),
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
            PlanningStep: [_make_plan_log(console)],
            ActionStep: [_make_step_log(console), _prune_old_steps],
        },
        final_answer_checks=[_check_answer],
    )
    return agent, tool_names
