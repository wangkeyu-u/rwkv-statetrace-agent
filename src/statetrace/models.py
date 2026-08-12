"""Shared, serialisable models used by the agent runtime.

The project deliberately keeps these models dependency-free.  Backends and user
interfaces can therefore import them without pulling in a validation framework.
Validation of untrusted model output belongs in :mod:`statetrace.protocol`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .backends.base import GenerationResult

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class AgentStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_FOR_MODEL = "WAITING_FOR_MODEL"
    EXECUTING_TOOL = "EXECUTING_TOOL"
    VALIDATING = "VALIDATING"
    REPLANNING = "REPLANNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class ErrorCode(StrEnum):
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    INVALID_JSON = "INVALID_JSON"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    PATH_OUTSIDE_WORKSPACE = "PATH_OUTSIDE_WORKSPACE"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    BINARY_FILE = "BINARY_FILE"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    COMMAND_NOT_ALLOWED = "COMMAND_NOT_ALLOWED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    DUPLICATE_TOOL_CALL = "DUPLICATE_TOOL_CALL"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_BACKEND_ERROR = "MODEL_BACKEND_ERROR"
    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
    CHECKPOINT_CORRUPTED = "CHECKPOINT_CORRUPTED"
    MODEL_STATE_MISMATCH = "MODEL_STATE_MISMATCH"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    TASK_TIMEOUT = "TASK_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: str
    arguments: dict[str, JSONValue]
    thought_summary: str = ""
    type: str = field(default="tool_call", init=False)

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "type": self.type,
            "thought_summary": self.thought_summary,
            "tool": self.tool,
            "arguments": self.arguments,
        }


@dataclass(frozen=True, slots=True)
class Finding:
    file: str
    line: int
    claim: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinalReport:
    summary: str
    findings: tuple[Finding, ...]
    verification: dict[str, JSONValue]
    recommendations: tuple[str, ...]
    uncertainties: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "summary": self.summary,
            "findings": [
                {
                    "file": finding.file,
                    "line": finding.line,
                    "claim": finding.claim,
                    "evidence_ids": list(finding.evidence_ids),
                }
                for finding in self.findings
            ],
            "verification": self.verification,
            "recommendations": list(self.recommendations),
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True, slots=True)
class FinalAction:
    report: FinalReport
    type: str = field(default="final", init=False)
    tool: str = field(default="finish_report", init=False)

    def as_dict(self) -> dict[str, JSONValue]:
        return {"type": self.type, "tool": self.tool, "arguments": self.report.as_dict()}


AgentAction = ToolCall | FinalAction


@dataclass(slots=True)
class Observation:
    status: str
    message: str = ""
    data: dict[str, JSONValue] = field(default_factory=dict)
    error_code: ErrorCode | None = None
    evidence_id: str | None = None
    truncated: bool = False
    duration_ms: float = 0.0
    type: str = field(default="observation", init=False)

    @property
    def ok(self) -> bool:
        return self.status in {"success", "warning"} and self.error_code is None

    def as_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "type": self.type,
            "status": self.status,
            "message": self.message,
            "data": self.data,
            "truncated": self.truncated,
            "duration_ms": self.duration_ms,
        }
        if self.error_code is not None:
            payload["error_code"] = self.error_code.value
        if self.evidence_id is not None:
            payload["evidence_id"] = self.evidence_id
        return payload


@dataclass(slots=True)
class AgentStep:
    number: int
    action: AgentAction | None = None
    observation: Observation | None = None
    generation: GenerationResult | None = None


@dataclass(slots=True)
class AgentTask:
    task_id: str
    goal: str
    workspace: Path
    status: AgentStatus = AgentStatus.CREATED
    step: int = 0
    model_state: Any = None
    steps: list[AgentStep] = field(default_factory=list)
    evidence: dict[str, Observation] = field(default_factory=dict)
    final_report: FinalReport | None = None
    validation_failures: int = 0

    def as_dict(self, *, include_model_state: bool = False) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "task_id": self.task_id,
            "goal": self.goal,
            "workspace": str(self.workspace),
            "status": self.status.value,
            "step": self.step,
            "evidence_ids": list(self.evidence),
            "tool_call_count": sum(1 for item in self.steps if isinstance(item.action, ToolCall)),
            "validation_failures": self.validation_failures,
            "final_report": self.final_report.as_dict() if self.final_report else None,
        }
        if include_model_state:
            payload["model_state"] = self.model_state
        return payload
