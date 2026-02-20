"""Shared infrastructure for agent tool modules."""

from collections.abc import Callable
from pathlib import Path

from rich.console import Console


class _ToolContext:
    """Shared context for tool functions. Set once at startup via ``set_tools_cwd``."""

    __slots__ = ("cwd", "console")

    def __init__(self) -> None:
        self.cwd = "."
        self.console: Console | None = None


_ctx = _ToolContext()


def set_tools_cwd(cwd: str) -> None:
    """Set the working directory used by tool functions."""
    _ctx.cwd = cwd


def set_tools_console(console: Console) -> None:
    """Set the live console used for dim output during tool execution."""
    _ctx.console = console


def _resolve_path(file_path: str, cwd: str | None = None) -> Path:
    """Resolve *file_path* relative to ``_ctx.cwd`` and guard against path traversal.

    Raises ``ValueError`` if the resolved path escapes the working directory.
    """
    root = Path(cwd or _ctx.cwd).resolve()
    resolved = (root / file_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes working directory: {file_path}")
    return resolved


def _make_capture_console() -> tuple[Console, Callable[[], str]]:
    """Create a Console that captures output.

    When a live console is available, output is printed dim in real time and
    recorded via ``Console(record=True)``.  The second return value is a
    callable that returns the captured text.
    """
    if _ctx.console is not None:
        con = Console(record=True, style="dim")
        return con, lambda: con.export_text()
    import io

    buf = io.StringIO()
    return Console(file=buf, no_color=True), buf.getvalue
