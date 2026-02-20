"""Tests for tools_filesystem.py — read, write, edit file tools."""

from aai_cli.tools_filesystem import edit_file, read_file, write_file

# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_success(tools_ctx):
    (tools_ctx / "hello.txt").write_text("line one\nline two\n")
    result = read_file("hello.txt")
    assert "line one" in result
    assert "line two" in result
    assert "1 |" in result
    assert "2 |" in result


def test_read_file_not_found(tools_ctx):
    result = read_file("nonexistent.txt")
    assert "Error" in result
    assert "not found" in result


def test_read_file_path_traversal(tools_ctx):
    result = read_file("../../etc/passwd")
    assert "Error" in result
    assert "escapes" in result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_file_creates(tools_ctx):
    result = write_file("new.txt", "hello world")
    assert "Wrote new.txt" in result
    assert (tools_ctx / "new.txt").read_text() == "hello world"


def test_write_file_creates_dirs(tools_ctx):
    result = write_file("sub/dir/file.txt", "content")
    assert "Wrote" in result
    assert (tools_ctx / "sub" / "dir" / "file.txt").read_text() == "content"


def test_write_file_path_traversal(tools_ctx):
    result = write_file("../../etc/hack", "bad")
    assert "Error" in result
    assert "escapes" in result


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def test_edit_file_success(tools_ctx):
    (tools_ctx / "code.py").write_text("x = 1\ny = 2\n")
    result = edit_file("code.py", "x = 1", "x = 42")
    assert "Edited" in result
    assert (tools_ctx / "code.py").read_text() == "x = 42\ny = 2\n"


def test_edit_file_not_found(tools_ctx):
    result = edit_file("missing.py", "old", "new")
    assert "Error" in result
    assert "not found" in result


def test_edit_file_old_string_missing(tools_ctx):
    (tools_ctx / "code.py").write_text("x = 1\n")
    result = edit_file("code.py", "not_here", "new")
    assert "Error" in result
    assert "not found" in result


def test_edit_file_multiple_matches_rejected(tools_ctx):
    (tools_ctx / "code.py").write_text("aaa\naaa\n")
    result = edit_file("code.py", "aaa", "bbb")
    assert "Error" in result
    assert "2 times" in result


def test_edit_file_replace_all(tools_ctx):
    (tools_ctx / "code.py").write_text("aaa\naaa\n")
    result = edit_file("code.py", "aaa", "bbb", replace_all=True)
    assert "Edited" in result
    assert "2 occurrence" in result
    assert (tools_ctx / "code.py").read_text() == "bbb\nbbb\n"


def test_edit_file_path_traversal(tools_ctx):
    result = edit_file("../../etc/passwd", "root", "hacked")
    assert "Error" in result
    assert "escapes" in result
