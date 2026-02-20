"""Tests for REPL utilities and slash commands."""

import io

from rich.console import Console

from aai_cli.repl import _COMMANDS, _cmd_clear, _cmd_help, _cmd_ideas, _extract_tool_calls


def _console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, no_color=True, width=200), buf


# ---------------------------------------------------------------------------
# _extract_tool_calls
# ---------------------------------------------------------------------------


def test_extract_tool_calls_single():
    code = 'result = read_file("foo.py")'
    assert _extract_tool_calls(code, {"read_file"}) == ["read_file"]


def test_extract_tool_calls_multiple():
    code = 'x = read_file("a.py")\ny = write_file("b.py", "content")'
    names = _extract_tool_calls(code, {"read_file", "write_file"})
    assert names == ["read_file", "write_file"]


def test_extract_tool_calls_no_match():
    code = 'x = some_other_func("a")'
    assert _extract_tool_calls(code, {"read_file"}) == []


def test_extract_tool_calls_syntax_error():
    code = "def broken(:"
    assert _extract_tool_calls(code, {"read_file"}) == []


def test_extract_tool_calls_nested():
    code = 'result = edit_file("f", read_file("g"), "new")'
    names = _extract_tool_calls(code, {"edit_file", "read_file"})
    assert "edit_file" in names
    assert "read_file" in names


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def test_cmd_help_output():
    con, buf = _console()
    _cmd_help(con)
    output = buf.getvalue()
    assert "/ideas" in output
    assert "/clear" in output
    assert "/help" in output


def test_cmd_ideas_output():
    con, buf = _console()
    _cmd_ideas(con)
    output = buf.getvalue()
    assert "voice agent" in output.lower() or "Voice" in output
    assert "Build" in output


def test_cmd_clear_prints_banner():
    con, buf = _console()
    _cmd_clear(con)
    output = buf.getvalue()
    assert "What would you like to build?" in output


# ---------------------------------------------------------------------------
# _COMMANDS dict
# ---------------------------------------------------------------------------


def test_commands_dict_has_all_commands():
    assert "/clear" in _COMMANDS
    assert "/help" in _COMMANDS
    assert "/ideas" in _COMMANDS
    assert len(_COMMANDS) == 3
