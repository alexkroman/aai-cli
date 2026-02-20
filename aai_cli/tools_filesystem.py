"""Agent tools for file operations."""

from smolagents import tool

from .tools import _resolve_path


@tool
def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact string match in a file with new content.

    Reads the file, finds old_string, replaces it with new_string, and writes back.
    By default old_string must appear exactly once to avoid ambiguous edits.
    Set replace_all=True to replace every occurrence (useful for renaming).

    Use this to make targeted edits to existing files without rewriting them entirely.

    Args:
        file_path: Path to the file to edit (relative to working directory).
        old_string: The exact text to find and replace.
        new_string: The replacement text.
        replace_all: If True, replace all occurrences. If False (default), old_string must be unique.
    """
    try:
        path = _resolve_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    if not path.is_file():
        return f"Error: {file_path} not found."

    content = path.read_text()
    count = content.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {file_path}."
    if not replace_all and count > 1:
        return f"Error: old_string appears {count} times in {file_path}. Use replace_all=True or provide a more specific match."

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)
    path.write_text(new_content)
    return f"Edited {file_path}: replaced {count} occurrence(s)."


@tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a file.

    Use this before editing a file to understand its current content and structure.

    Args:
        file_path: Path to the file to read (relative to working directory).
    """
    try:
        path = _resolve_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    if not path.is_file():
        return f"Error: {file_path} not found."
    content = path.read_text()
    lines = content.splitlines()
    numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


@tool
def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Use this to create new files from scratch. Parent directories are created
    automatically. For targeted changes to existing files, prefer edit_file instead.

    Args:
        file_path: Path to write to (relative to working directory).
        content: The full file content to write.
    """
    try:
        path = _resolve_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"Wrote {file_path} ({len(content)} bytes)."
