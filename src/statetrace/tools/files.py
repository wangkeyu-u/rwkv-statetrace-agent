"""Safe, bounded directory listing and text-file reading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ..models import ErrorCode, JSONValue, Observation
from .base import Tool, ToolContext, ToolError, optional_int, require_string, truncate

IGNORED_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
IGNORED_SUFFIXES = {".bin", ".gguf", ".onnx", ".pth", ".safetensors"}


class ListFilesTool(Tool):
    name = "list_files"
    description = "List sorted files beneath a workspace directory with bounded depth."
    schema: dict[str, JSONValue] = {
        "path": "string (default '.')",
        "max_depth": "integer 0..6 (default 2)",
        "max_files": "integer 1..500 (default 200)",
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> Observation:
        path_value = arguments.get("path", ".")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "path must be a non-empty string.")
        max_depth = optional_int(arguments, "max_depth", default=2, minimum=0, maximum=6)
        max_files = optional_int(arguments, "max_files", default=200, minimum=1, maximum=500)
        target = context.resolve_path(path_value, must_exist=True)
        if not target.is_dir():
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "list_files path must be a directory.")

        found: list[str] = []

        def visit(directory: Path, depth: int) -> None:
            if len(found) >= max_files:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda item: item.name.lower())
            except OSError as exc:
                raise ToolError(ErrorCode.FILE_NOT_FOUND, f"Cannot read directory: {exc}") from exc
            for child in children:
                if len(found) >= max_files:
                    return
                if child.name in IGNORED_NAMES or child.suffix.lower() in IGNORED_SUFFIXES:
                    continue
                if child.is_symlink():
                    # Symlinks are omitted: even a link below root can point outside it.
                    continue
                if child.is_file():
                    found.append(child.relative_to(context.root).as_posix())
                elif child.is_dir() and depth < max_depth:
                    visit(child, depth + 1)

        visit(target, 0)
        return Observation(
            status="success",
            message=f"Listed {len(found)} files.",
            data={"files": cast(JSONValue, found), "truncated": len(found) >= max_files},
            truncated=len(found) >= max_files,
        )


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file with stable 1-based line numbers."
    schema: dict[str, JSONValue] = {
        "path": "string",
        "start_line": "integer >= 1 (default 1)",
        "end_line": "integer >= start_line, at most 200 lines",
    }

    def __init__(self, max_lines: int = 200) -> None:
        self.max_lines = max_lines

    def run(self, arguments: dict[str, Any], context: ToolContext) -> Observation:
        supplied = require_string(arguments, "path")
        start = optional_int(arguments, "start_line", default=1, minimum=1, maximum=10_000_000)
        raw_end = arguments.get("end_line", start + self.max_lines - 1)
        if isinstance(raw_end, bool) or not isinstance(raw_end, int) or raw_end < start:
            raise ToolError(
                ErrorCode.INVALID_ARGUMENTS, "end_line must be an integer >= start_line."
            )
        if raw_end - start + 1 > self.max_lines:
            raise ToolError(
                ErrorCode.INVALID_ARGUMENTS,
                f"A read may contain at most {self.max_lines} lines.",
            )
        path = context.resolve_path(supplied, must_exist=True)
        if not path.is_file():
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "read_file path must be a file.")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ToolError(ErrorCode.FILE_NOT_FOUND, f"Cannot read file: {exc}") from exc
        if b"\x00" in raw[:8192]:
            raise ToolError(ErrorCode.BINARY_FILE, "Binary files cannot be read with read_file.")
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ToolError(ErrorCode.BINARY_FILE, "File is not valid UTF-8 text.") from exc

        selected = lines[start - 1 : raw_end]
        rendered = "\n".join(f"{number:>6} | {line}" for number, line in enumerate(selected, start))
        rendered, was_truncated = truncate(rendered, context.max_output_chars)
        return Observation(
            status="success",
            message=f"Read lines {start}-{min(raw_end, len(lines))} from {supplied}.",
            data={
                "path": path.relative_to(context.root).as_posix(),
                "start_line": start,
                "end_line": min(raw_end, len(lines)),
                "total_lines": len(lines),
                "content": rendered,
            },
            truncated=was_truncated,
        )
