"""Allow-listed test execution without a shell."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from ..models import ErrorCode, JSONValue, Observation
from .base import Tool, ToolContext, ToolError, optional_int, require_string, truncate

FORBIDDEN_TOKENS = {";", "&&", "||", "|", ">", ">>", "<", "2>", "&"}


class RunTestsTool(Tool):
    name = "run_tests"
    description = "Run an allow-listed test command in the workspace, without a shell."
    schema: dict[str, JSONValue] = {
        "command": "pytest / python -m pytest / npm test / npm run test",
        "timeout_seconds": "integer 1..60 (default 30)",
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> Observation:
        command_text = require_string(arguments, "command")
        timeout = optional_int(arguments, "timeout_seconds", default=30, minimum=1, maximum=60)
        if "$(" in command_text or "`" in command_text or "\n" in command_text:
            raise ToolError(ErrorCode.COMMAND_NOT_ALLOWED, "Command substitution is not allowed.")
        try:
            argv = shlex.split(command_text, posix=True)
        except ValueError as exc:
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, f"Invalid command quoting: {exc}") from exc
        if not argv or any(token in FORBIDDEN_TOKENS for token in argv):
            raise ToolError(ErrorCode.COMMAND_NOT_ALLOWED, "Shell operators are not allowed.")
        if not self._allowed(argv):
            raise ToolError(
                ErrorCode.COMMAND_NOT_ALLOWED,
                "Allowed commands are pytest, python -m pytest, npm test, and npm run test.",
            )
        self._validate_arguments(argv, context)
        # Use the currently running interpreter so virtual environments and
        # systems without a `python` shim behave consistently, without a shell.
        if argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        try:
            completed = subprocess.run(
                argv,
                cwd=context.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            raise ToolError(
                ErrorCode.TOOL_TIMEOUT,
                f"Test command timed out after {timeout} seconds.",
                {"stdout": stdout, "stderr": stderr},
            ) from exc
        stdout, stdout_cut = truncate(completed.stdout, context.max_output_chars // 2)
        stderr, stderr_cut = truncate(completed.stderr, context.max_output_chars // 2)
        if "No module named pytest" in stderr:
            raise ToolError(
                ErrorCode.INTERNAL_ERROR,
                "pytest is not installed in the active environment; install the dev extra before running tests.",
                {"command": command_text, "exit_code": completed.returncode, "stderr": stderr},
            )
        return Observation(
            status="success",
            message=f"Test process exited with code {completed.returncode}.",
            data={
                "command": cast(JSONValue, argv),
                "exit_code": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
            truncated=stdout_cut or stderr_cut,
        )

    @staticmethod
    def _allowed(argv: list[str]) -> bool:
        if argv[0] == "pytest":
            return True
        if len(argv) >= 3 and argv[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
            return True
        if argv[:2] == ["npm", "test"]:
            return True
        return len(argv) >= 3 and argv[:3] == ["npm", "run", "test"]

    @staticmethod
    def _validate_arguments(argv: list[str], context: ToolContext) -> None:
        """Reject arguments which can redirect pytest beyond the workspace.

        This is a guardrail, not a sandbox: repository tests execute repository
        code. The README therefore restricts this tool to trusted fixtures.
        """

        if argv[0] == "npm":
            if argv not in (["npm", "test"], ["npm", "run", "test"]):
                raise ToolError(
                    ErrorCode.COMMAND_NOT_ALLOWED,
                    "npm test commands do not accept additional arguments in this runtime.",
                )
            return
        offset = 3 if argv[0] in {"python", "python3"} else 1
        dangerous_options = (
            "--rootdir",
            "--confcutdir",
            "--basetemp",
            "--override-ini",
            "--import-mode",
        )
        args = argv[offset:]
        for index, item in enumerate(args):
            if item == "-c" or item == "-p" or item.startswith(dangerous_options):
                raise ToolError(
                    ErrorCode.COMMAND_NOT_ALLOWED,
                    f"pytest option {item!r} is not allowed by the bounded runner.",
                )
            if item.startswith("-"):
                continue
            # Values following selection/output flags are not filesystem paths.
            if index and args[index - 1] in {"-k", "-m", "--tb", "--maxfail"}:
                continue
            supplied = Path(item.split("::", 1)[0])
            if supplied.is_absolute() or ".." in supplied.parts:
                try:
                    context.resolve_path(str(supplied), must_exist=False)
                except ToolError as exc:
                    raise ToolError(
                        ErrorCode.COMMAND_NOT_ALLOWED,
                        "Test targets must stay inside the task workspace.",
                    ) from exc
