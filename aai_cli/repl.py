"""Interactive REPL loop and slash-command dispatch."""

import ast
import warnings
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from smolagents import CodeAgent
from smolagents.memory import FinalAnswerStep, ToolCall

from .agent import create_agent
from .service import get_env_key
from .tools import set_tools_console, set_tools_cwd

_BANNER = (
    "\U0001f43b What would you like to build? "
    "Type a message to start. /ideas for inspiration. /help for help.\n"
)


def _cmd_clear(console: Console) -> None:
    console.clear()
    console.print(_BANNER, style="dim")


def _cmd_help(console: Console) -> None:
    console.print(
        Markdown(
            "## Commands\n"
            "- **/ideas** — sample apps to build\n"
            "- **/clear** — clear conversation context\n"
            "- **/help** — show this help\n"
            "- **Ctrl+C** — interrupt current operation or exit\n"
        )
    )


def _cmd_ideas(console: Console) -> None:
    console.print()
    console.print(
        Markdown(
            "## Build\n"
            "- Create a simple voice agent\n"
            "- An action item extractor from meeting recordings\n"
            "- A podcast chapter generator from audio files\n"
            "- A sentiment analysis pipeline for customer calls\n"
            "- A real-time transcription app with speaker labels\n"
            "- A content moderation system for spoken audio\n"
            "\n## Evaluate\n"
            "- Measure transcription quality (WER) against any Hugging Face dataset\n"
            "\n## Optimize\n"
            "- Find the best transcription prompt for Universal-3-Pro using LLM-guided optimization\n"
        )
    )
    console.print()


def _extract_tool_calls(code: str, tool_names: set[str]) -> list[str]:
    """Extract tool function names from code using AST parsing."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in tool_names
        ):
            names.append(node.func.id)
    return names


def _run_streaming(
    agent: CodeAgent, msg: str, console: Console, tool_names: set[str], reset: bool = True
) -> str | None:
    """Run agent with streaming to show tool names before execution."""
    result = None
    for event in agent.run(msg, stream=True, reset=reset):  # type: ignore[reportGeneralTypeIssues]
        if isinstance(event, ToolCall) and event.name == "python_interpreter":
            code = event.arguments if isinstance(event.arguments, str) else ""
            for name in _extract_tool_calls(code, tool_names):
                console.print(f"\n⏺ {name}", style="dim")
        if isinstance(event, FinalAnswerStep):
            result = event.output
    return result


_COMMANDS: dict[str, Callable[[Console], None]] = {
    "/clear": _cmd_clear,
    "/help": _cmd_help,
    "/ideas": _cmd_ideas,
}


def run_agent(extra_args: list[str] | None = None) -> None:
    """Run the interactive coding agent loop."""
    cwd = str(Path.cwd())
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    console = Console()
    try:
        anthropic_key = get_env_key("ANTHROPIC_API_KEY")
    except ValueError as e:
        console.print(f"  {e}", style="bold red")
        return

    set_tools_cwd(cwd)
    set_tools_console(console)
    console.clear()

    agent, tool_names_list = create_agent(anthropic_key, console, cwd)
    tool_names = set(tool_names_list)

    def _safe_run(msg: str, reset: bool) -> bool:
        """Run agent, print result, return whether first_turn should become False."""
        try:
            result = _run_streaming(agent, msg, console, tool_names, reset=reset)
            if result:
                console.print()
                console.print(Markdown(str(result)))
                console.print()
            return True
        except KeyboardInterrupt:
            console.print("\n  Interrupted.", style="dim")
        except Exception as e:
            console.print(f"  Error: {e}", style="bold red")
        return False

    first_turn = True

    if extra_args:
        first_msg = " ".join(extra_args)
        console.print(f"  > {first_msg}", style="dim")
        if _safe_run(first_msg, reset=True):
            first_turn = False
    else:
        console.print(_BANNER, style="dim")

    history_path = Path.home() / ".aai_history"
    session: PromptSession = PromptSession(history=FileHistory(str(history_path)))

    while True:
        try:
            user_input = session.prompt("> ")
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye.", style="dim")
            break
        if not user_input.strip():
            continue
        cmd = user_input.strip()

        # Slash-command dispatch
        handler = _COMMANDS.get(cmd)
        if handler is not None:
            if cmd == "/clear":
                first_turn = True
            handler(console)
            continue

        if _safe_run(user_input, reset=first_turn):
            first_turn = False
