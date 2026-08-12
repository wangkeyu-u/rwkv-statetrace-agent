"""Source search backed by ripgrep and safely bounded arguments."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, cast

from ..models import ErrorCode, JSONValue, Observation
from .base import Tool, ToolContext, ToolError, optional_int, require_string, truncate


class SearchCodeTool(Tool):
    name = "search_code"
    description = "Search workspace text files and return matching paths, lines, and text."
    schema: dict[str, JSONValue] = {
        "query": "literal string",
        "path": "string (default '.')",
        "glob": "optional glob such as '*.py'",
        "max_results": "integer 1..100 (default 30)",
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> Observation:
        query = require_string(arguments, "query")
        supplied = arguments.get("path", ".")
        if not isinstance(supplied, str) or not supplied.strip():
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "path must be a non-empty string.")
        target = context.resolve_path(supplied, must_exist=True)
        glob = arguments.get("glob")
        if glob is not None and (not isinstance(glob, str) or not glob.strip()):
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "glob must be a non-empty string.")
        maximum = optional_int(arguments, "max_results", default=30, minimum=1, maximum=100)
        executable = shutil.which("rg")
        if executable is None:
            raise ToolError(ErrorCode.INTERNAL_ERROR, "ripgrep (rg) is required for search_code.")
        command = [
            executable,
            "--line-number",
            "--column",
            "--no-heading",
            "--color=never",
            "--fixed-strings",
            "--hidden",
            "--glob=!.git/**",
        ]
        if glob:
            command.extend(["--glob", glob])
        command.extend(["--", query, str(target)])
        try:
            completed = subprocess.run(
                command,
                cwd=context.root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(ErrorCode.TOOL_TIMEOUT, "search_code timed out after 10 seconds.") from exc
        if completed.returncode not in {0, 1}:
            raise ToolError(ErrorCode.INTERNAL_ERROR, f"rg failed: {completed.stderr.strip()}")
        raw_lines = completed.stdout.splitlines()
        selected = raw_lines[:maximum]
        results: list[dict[str, JSONValue]] = []
        for match in selected:
            pieces = match.split(":", 3)
            if len(pieces) != 4:
                continue
            raw_path, line, column, content = pieces
            try:
                relative = context.resolve_path(raw_path, must_exist=True).relative_to(context.root).as_posix()
                results.append(
                    {"path": relative, "line": int(line), "column": int(column), "text": content}
                )
            except (ToolError, ValueError):
                continue
        # Also enforce the global character budget.
        rendered, char_truncated = truncate(str(results), context.max_output_chars)
        if char_truncated:
            # Preserve structured data by dropping matches until it fits.
            while results and len(str(results)) > context.max_output_chars:
                results.pop()
        was_truncated = len(raw_lines) > len(results) or char_truncated
        return Observation(
            status="success",
            message=f"Found {len(results)} matching lines.",
            data={"results": cast(JSONValue, results), "truncated": was_truncated},
            truncated=was_truncated,
        )
