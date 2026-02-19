"""Interactive REPL loop and slash-command dispatch."""

import os
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from smolagents.memory import FinalAnswerStep, ToolCall

from .agent import TOOL_NAMES, create_agent
from .tools import init_tools

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


def _run_streaming(agent, msg: str, console: Console, reset: bool = True) -> str | None:
    """Run agent with streaming to show tool names before execution."""
    result = None
    for event in agent.run(msg, stream=True, reset=reset):
        if isinstance(event, ToolCall) and event.name == "python_interpreter":
            code = event.arguments if isinstance(event.arguments, str) else ""
            for name in TOOL_NAMES:
                if name in code:
                    console.print(f"\n⏺ {name}", style="dim")
                    break
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

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    console = Console()

    if not anthropic_key:
        console.print("  Missing env var: ANTHROPIC_API_KEY", style="bold red")
        sys.exit(1)

    init_tools(console, cwd)
    console.clear()

    agent = create_agent(anthropic_key, console, cwd)

    first_turn = True

    if extra_args:
        first_msg = " ".join(extra_args)
        console.print(f"  > {first_msg}", style="dim")
        try:
            result = _run_streaming(agent, first_msg, console)
            if result:
                console.print()
                console.print(Markdown(str(result)))
                console.print()
        except KeyboardInterrupt:
            console.print("\n  Interrupted.", style="dim")
        except Exception as e:
            console.print(f"  Error: {e}", style="bold red")
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

        try:
            result = _run_streaming(agent, user_input, console, reset=first_turn)
            if result:
                console.print()
                console.print(Markdown(str(result)))
                console.print()
            first_turn = False
        except KeyboardInterrupt:
            console.print("\n  Interrupted.", style="dim")
        except Exception as e:
            console.print(f"  Error: {e}", style="bold red")
