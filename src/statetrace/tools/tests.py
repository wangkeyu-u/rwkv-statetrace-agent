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
PYTEST_FLAG_OPTIONS = {
    "-q",
    "--quiet",
    "-v",
    "--verbose",
    "-x",
    "--exitfirst",
    "--lf",
    "--last-failed",
    "--ff",
    "--failed-first",
    "--nf",
    "--new-first",
    "--disable-warnings",
    "--strict-markers",
    "--strict-config",
    "-s",
}
PYTEST_VALUE_OPTIONS = {"-k", "-m", "--tb", "--maxfail", "--color"}
PYTEST_TB_VALUES = {"auto", "long", "short", "line", "native", "no"}
PYTEST_COLOR_VALUES = {"yes", "no", "auto"}


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
        args = argv[offset:]
        index = 0
        positional_only = False
        while index < len(args):
            item = args[index]
            if item == "--":
                positional_only = True
                index += 1
                continue
            if not positional_only and item in PYTEST_FLAG_OPTIONS:
                index += 1
                continue
            if not positional_only and item.startswith("-"):
                option, separator, inline_value = item.partition("=")
                if option not in PYTEST_VALUE_OPTIONS:
                    raise ToolError(
                        ErrorCode.COMMAND_NOT_ALLOWED,
                        f"pytest option {item!r} is not in the bounded runner allow-list.",
                    )
                if separator:
                    value = inline_value
                else:
                    index += 1
                    if index >= len(args):
                        raise ToolError(
                            ErrorCode.INVALID_ARGUMENTS,
                            f"pytest option {option!r} requires a value.",
                        )
                    value = args[index]
                RunTestsTool._validate_option_value(option, value)
                index += 1
                continue
            supplied = Path(item.split("::", 1)[0])
            try:
                context.resolve_path(str(supplied), must_exist=False)
            except ToolError as exc:
                raise ToolError(
                    ErrorCode.COMMAND_NOT_ALLOWED,
                    "Test targets must stay inside the task workspace.",
                ) from exc
            index += 1

    @staticmethod
    def _validate_option_value(option: str, value: str) -> None:
        if not value:
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, f"pytest option {option!r} requires a value.")
        if option == "--tb" and value not in PYTEST_TB_VALUES:
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, f"Unsupported --tb value: {value!r}.")
        if option == "--color" and value not in PYTEST_COLOR_VALUES:
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, f"Unsupported --color value: {value!r}.")
        if option == "--maxfail" and (not value.isdigit() or int(value) < 1):
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "--maxfail must be a positive integer.")
