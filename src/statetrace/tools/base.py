"""Tool contracts and shared safety utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import ErrorCode, JSONValue, Observation


@dataclass(frozen=True, slots=True)
class ToolContext:
    workspace: Path
    max_output_chars: int = 20_000

    @property
    def root(self) -> Path:
        return self.workspace.expanduser().resolve()

    def resolve_path(self, supplied: str, *, must_exist: bool = False) -> Path:
        if not isinstance(supplied, str) or not supplied.strip():
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "path must be a non-empty string.")
        candidate = (self.root / supplied).resolve() if not Path(supplied).is_absolute() else Path(supplied).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ToolError(
                ErrorCode.PATH_OUTSIDE_WORKSPACE,
                "The requested path is outside the task workspace.",
            ) from exc
        if must_exist and not candidate.exists():
            raise ToolError(ErrorCode.FILE_NOT_FOUND, f"Path does not exist: {supplied}")
        return candidate


class ToolError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, data: dict[str, JSONValue] | None = None):
        super().__init__(message)
        self.code = code
        self.data = data or {}

    def as_observation(self) -> Observation:
        return Observation(
            status="error", message=str(self), error_code=self.code, data=self.data
        )


class Tool(ABC):
    name: str
    description: str
    schema: dict[str, JSONValue]

    @abstractmethod
    def run(self, arguments: dict[str, Any], context: ToolContext) -> Observation: ...

    def describe(self) -> dict[str, JSONValue]:
        return {"name": self.name, "description": self.description, "arguments": self.schema}


def require_string(arguments: dict[str, Any], name: str, *, default: str | None = None) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(ErrorCode.INVALID_ARGUMENTS, f"{name} must be a non-empty string.")
    return value.strip()


def optional_int(
    arguments: dict[str, Any], name: str, *, default: int, minimum: int, maximum: int
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ToolError(
            ErrorCode.INVALID_ARGUMENTS,
            f"{name} must be an integer between {minimum} and {maximum}.",
        )
    return value


def truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = "\n... [output truncated]"
    return text[: max(0, limit - len(marker))] + marker, True
