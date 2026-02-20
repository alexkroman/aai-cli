"""Tests for tools.py shared infrastructure."""

import io

import pytest
from rich.console import Console

from aai_cli.tools import (
    _ctx,
    _make_capture_console,
    _resolve_path,
    set_tools_console,
    set_tools_cwd,
)

# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------


def test_resolve_path_relative(tmp_path):
    result = _resolve_path("foo.txt", cwd=str(tmp_path))
    assert result == (tmp_path / "foo.txt").resolve()


def test_resolve_path_absolute_within(tmp_path):
    target = tmp_path / "sub" / "file.txt"
    result = _resolve_path(str(target), cwd=str(tmp_path))
    assert result == target.resolve()


def test_resolve_path_escapes_raises(tmp_path):
    with pytest.raises(ValueError, match="Path escapes working directory"):
        _resolve_path("../../etc/passwd", cwd=str(tmp_path))


def test_resolve_path_uses_ctx_fallback(tools_ctx):
    """When no cwd arg is passed, should use _ctx.cwd."""
    result = _resolve_path("test.txt")
    assert result == (tools_ctx / "test.txt").resolve()


def test_resolve_path_nested_relative(tmp_path):
    result = _resolve_path("sub/dir/file.py", cwd=str(tmp_path))
    assert result == (tmp_path / "sub" / "dir" / "file.py").resolve()


# ---------------------------------------------------------------------------
# set_tools_cwd / set_tools_console
# ---------------------------------------------------------------------------


def test_set_tools_cwd(tools_ctx):
    set_tools_cwd("/tmp/test")
    assert _ctx.cwd == "/tmp/test"


def test_set_tools_console(tools_ctx):
    con = Console()
    set_tools_console(con)
    assert _ctx.console is con


# ---------------------------------------------------------------------------
# _make_capture_console
# ---------------------------------------------------------------------------


def test_make_capture_console_without_live(tools_ctx):
    """When _ctx.console is None, should capture to StringIO."""
    _ctx.console = None
    con, get_text = _make_capture_console()
    con.print("hello")
    output = get_text()
    assert "hello" in output


def test_make_capture_console_with_live(tools_ctx):
    """When _ctx.console is set, should use record-mode Console."""
    _ctx.console = Console(file=io.StringIO())
    con, get_text = _make_capture_console()
    con.print("recorded")
    output = get_text()
    assert "recorded" in output
