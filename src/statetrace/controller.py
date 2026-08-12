"""Strict model → action → observation agent loop."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends.base import GenerationResult, ModelBackend, SamplingConfig
from .context import build_incremental_prompt, build_prompt
from .models import (
    AgentStatus,
    AgentStep,
    AgentTask,
    ErrorCode,
    FinalAction,
    Observation,
    ToolCall,
)
from .protocol import ProtocolError, parse_action
from .tools.base import ToolContext
from .tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    max_steps: int = 12
    max_duplicate_calls: int = 2
    task_timeout_seconds: int = 300
    max_output_chars: int = 20_000
    sampling: SamplingConfig = SamplingConfig()

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_duplicate_calls < 1:
            raise ValueError("max_duplicate_calls must be >= 1")
        if self.task_timeout_seconds < 1:
            raise ValueError("task_timeout_seconds must be >= 1")


class AgentController:
    """Run a bounded agent while preserving model autonomy over each next action.

    ``validator`` may be a callable or an object exposing ``validate``.  It is
    deterministic and receives the final report plus structured evidence.
    ``trace`` and ``checkpoint_manager`` are optional duck-typed collaborators,
    keeping the core independently testable.
    """

    def __init__(
        self,
        *,
        backend: ModelBackend,
        tools: ToolRegistry,
        validator: Any | None = None,
        trace: Any | None = None,
        checkpoint_manager: Any | None = None,
        config: ControllerConfig | None = None,
        prompt_builder: Callable[[AgentTask, list[dict[str, Any]]], str] = build_prompt,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.validator = validator
        self.trace = trace
        self.checkpoints = checkpoint_manager
        self.config = config or ControllerConfig()
        self.prompt_builder = prompt_builder
        self._fingerprints: dict[str, int] = {}

    def new_task(self, goal: str, workspace: str | Path, task_id: str | None = None) -> AgentTask:
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"workspace is not a directory: {root}")
        if not goal.strip():
            raise ValueError("goal must be non-empty")
        task = AgentTask(
            task_id=task_id or f"task-{uuid.uuid4().hex[:12]}",
            goal=goal.strip(),
            workspace=root,
        )
        self._record("task_created", task_id=task.task_id, goal=task.goal, workspace=root)
        return task

    def restore_task(self, loaded_checkpoint: Any) -> AgentTask:
        """Rehydrate a task including prior actions and observations.

        The recurrent state remains backend-owned and comes from
        ``CheckpointManager.load``.  Structured task history is restored
        separately so prompt construction, evidence validation, and duplicate
        detection remain correct without replaying completed tools.
        """

        raw = loaded_checkpoint.task_state
        workspace = Path(raw["workspace"]).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"checkpoint workspace is not a directory: {workspace}")
        task = AgentTask(
            task_id=str(raw["task_id"]),
            goal=str(raw["goal"]),
            workspace=workspace,
            status=AgentStatus(str(raw.get("status", AgentStatus.INTERRUPTED.value))),
            step=int(raw.get("step", 0)),
            model_state=loaded_checkpoint.model_state,
            validation_failures=int(raw.get("validation_failures", 0)),
        )
        for raw_step in raw.get("steps", []):
            action = None
            if isinstance(raw_step.get("action"), dict):
                action = parse_action(json.dumps(raw_step["action"], ensure_ascii=False))
            raw_observation = raw_step.get("observation")
            observation = None
            if isinstance(raw_observation, dict):
                raw_code = raw_observation.get("error_code")
                try:
                    code = ErrorCode(raw_code) if raw_code else None
                except ValueError:
                    code = ErrorCode.INTERNAL_ERROR
                observation = Observation(
                    status=str(raw_observation.get("status", "error")),
                    message=str(raw_observation.get("message", "")),
                    data=raw_observation.get("data", {})
                    if isinstance(raw_observation.get("data", {}), dict)
                    else {},
                    error_code=code,
                    evidence_id=raw_observation.get("evidence_id"),
                    truncated=bool(raw_observation.get("truncated", False)),
                    duration_ms=float(raw_observation.get("duration_ms", 0.0)),
                )
            raw_generation = raw_step.get("generation")
            generation = None
            if isinstance(raw_generation, dict):
                generation = GenerationResult(
                    text=str(raw_generation.get("text", "")),
                    state=None,
                    prompt_tokens=int(raw_generation.get("prompt_tokens", 0)),
                    generated_tokens=int(raw_generation.get("generated_tokens", 0)),
                    duration_ms=float(raw_generation.get("duration_ms", 0.0)),
                    backend_name=str(raw_generation.get("backend_name", "unknown")),
                    model_name=str(raw_generation.get("model_name", "unknown")),
                )
            step = AgentStep(int(raw_step["number"]), action, observation, generation)
            task.steps.append(step)
            if observation and observation.evidence_id:
                task.evidence[observation.evidence_id] = observation
            if isinstance(action, ToolCall) and action.tool != "__protocol__":
                fingerprint = json.dumps(
                    {"tool": action.tool, "arguments": action.arguments},
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                self._fingerprints[fingerprint] = self._fingerprints.get(fingerprint, 0) + 1
            if isinstance(action, FinalAction) and observation is None:
                task.final_report = action.report
        self._record(
            "task_restored", task_id=task.task_id, step=task.step, checkpoint=str(loaded_checkpoint.metadata.path)
        )
        return task

    def run(self, task: AgentTask) -> AgentTask:
        if task.status == AgentStatus.COMPLETED:
            return task
        started = time.monotonic()
        task.status = AgentStatus.RUNNING
        if task.step == 0 and not task.steps:
            # A task-only step-zero checkpoint makes startup interruption
            # recoverable even before the first model response is available.
            self._save(task)
        while task.step < self.config.max_steps:
            if time.monotonic() - started >= self.config.task_timeout_seconds:
                task.status = AgentStatus.FAILED
                self._record(
                    "task_failed", task_id=task.task_id, status=ErrorCode.TASK_TIMEOUT.value
                )
                return task
            task.step += 1
            task.status = AgentStatus.WAITING_FOR_MODEL
            if (
                task.model_state is not None
                and getattr(self.backend, "context_mode", "full_transcript") == "incremental_state"
            ):
                prompt = build_incremental_prompt(task)
            else:
                prompt = self.prompt_builder(task, self.tools.descriptions())
            self._record(
                "model_request",
                task_id=task.task_id,
                step=task.step,
                backend=getattr(self.backend, "name", "unknown"),
                model_name=getattr(self.backend, "model_name", "unknown"),
            )
            try:
                generation = self.backend.generate(prompt, task.model_state, self.config.sampling)
                task.model_state = generation.state
            except Exception as exc:
                observation = Observation(
                    status="error",
                    error_code=ErrorCode.MODEL_BACKEND_ERROR,
                    message=f"Model backend failed: {type(exc).__name__}: {exc}",
                )
                task.steps.append(AgentStep(task.step, observation=observation))
                task.status = AgentStatus.FAILED
                self._record(
                    "task_failed",
                    task_id=task.task_id,
                    step=task.step,
                    status=ErrorCode.MODEL_BACKEND_ERROR.value,
                )
                return task
            self._record(
                "model_response",
                task_id=task.task_id,
                step=task.step,
                generated_tokens=generation.generated_tokens,
                prompt_tokens=generation.prompt_tokens,
                duration_ms=generation.duration_ms,
                backend=generation.backend_name,
                model_name=generation.model_name,
            )

            try:
                action = parse_action(generation.text)
            except ProtocolError as exc:
                observation = exc.as_observation()
                # A visible synthetic action ensures context builders include the
                # feedback while making clear that no environment tool ran.
                action = ToolCall(
                    tool="__protocol__",
                    arguments={"invalid_output": generation.text[:2000]},
                    thought_summary="Model output could not be parsed.",
                )
                task.steps.append(AgentStep(task.step, action, observation, generation))
                self._record_step(task, action, observation)
                self._save(task)
                continue

            if isinstance(action, FinalAction):
                if self._handle_final(task, action, generation):
                    return task
                self._save(task)
                continue

            observation = self._execute_action(task, action)
            task.steps.append(AgentStep(task.step, action, observation, generation))
            if observation.evidence_id:
                task.evidence[observation.evidence_id] = observation
            self._record_step(task, action, observation)
            self._save(task)

        task.status = AgentStatus.FAILED
        terminal = Observation(
            status="error",
            error_code=ErrorCode.MAX_STEPS_REACHED,
            message=f"Agent reached the maximum of {self.config.max_steps} model decisions.",
        )
        task.steps.append(AgentStep(task.step + 1, observation=terminal))
        self._record(
            "task_failed", task_id=task.task_id, status=ErrorCode.MAX_STEPS_REACHED.value
        )
        return task

    def _execute_action(self, task: AgentTask, action: ToolCall) -> Observation:
        fingerprint = json.dumps(
            {"tool": action.tool, "arguments": action.arguments},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        count = self._fingerprints.get(fingerprint, 0) + 1
        self._fingerprints[fingerprint] = count
        if count > self.config.max_duplicate_calls:
            return Observation(
                status="error",
                error_code=ErrorCode.DUPLICATE_TOOL_CALL,
                message="This exact tool call already ran. Use its evidence or change the arguments.",
                data={"tool": action.tool, "repeat_count": count},
            )
        task.status = AgentStatus.EXECUTING_TOOL
        observation = self.tools.execute(
            action.tool,
            action.arguments,
            ToolContext(task.workspace, max_output_chars=self.config.max_output_chars),
        )
        # Successful environment reads become citable evidence. Protocol,
        # unknown-tool, and duplicate errors are still observations but not facts.
        if observation.status == "success":
            observation.evidence_id = f"obs-{len(task.evidence) + 1:03d}"
        return observation

    def _handle_final(self, task: AgentTask, action: FinalAction, generation: Any) -> bool:
        task.status = AgentStatus.VALIDATING
        if self.validator is None:
            passed, feedback = True, None
        else:
            evidence = self._validator_evidence(task)
            method = getattr(self.validator, "validate", self.validator)
            result = method(action.report.as_dict(), evidence)
            passed = bool(getattr(result, "passed", result is True))
            feedback = getattr(result, "as_observation", lambda: result)()
        if passed:
            task.final_report = action.report
            task.status = AgentStatus.COMPLETED
            task.steps.append(AgentStep(task.step, action, None, generation))
            self._record("validation_passed", task_id=task.task_id, step=task.step)
            self._record("task_completed", task_id=task.task_id, step=task.step)
            self._save(task)
            return True
        task.validation_failures += 1
        task.status = AgentStatus.REPLANNING
        observation = Observation(
            status="validation_failed",
            error_code=ErrorCode.VALIDATION_FAILED,
            message="Final report failed deterministic validation. Collect evidence or correct it.",
            data=feedback if isinstance(feedback, dict) else {"details": str(feedback)},
        )
        task.steps.append(AgentStep(task.step, action, observation, generation))
        self._record(
            "validation_failed",
            task_id=task.task_id,
            step=task.step,
            feedback=observation.data,
        )
        return False

    def _validator_evidence(self, task: AgentTask) -> dict[str, dict[str, Any]]:
        evidence: dict[str, dict[str, Any]] = {}
        for step in task.steps:
            if not isinstance(step.action, ToolCall) or step.observation is None:
                continue
            evidence_id = step.observation.evidence_id
            if evidence_id:
                data = step.observation.data
                evidence[evidence_id] = {
                    "id": evidence_id,
                    "tool": step.action.tool,
                    "arguments": step.action.arguments,
                    "status": step.observation.status,
                    "artifact": {
                        "file": data.get("path"),
                        "start_line": data.get("start_line"),
                        "end_line": data.get("end_line"),
                    },
                    "data": data,
                }
        return evidence

    def _record_step(self, task: AgentTask, action: ToolCall, observation: Observation) -> None:
        self._record(
            "tool_call",
            task_id=task.task_id,
            step=task.step,
            tool=action.tool,
            arguments=action.arguments,
        )
        self._record(
            "tool_result",
            task_id=task.task_id,
            step=task.step,
            tool=action.tool,
            status=observation.status,
            error_code=observation.error_code.value if observation.error_code else None,
            evidence_id=observation.evidence_id,
            duration_ms=observation.duration_ms,
            truncated=observation.truncated,
        )

    def _save(self, task: AgentTask) -> None:
        if self.checkpoints is None:
            return
        metadata = self.checkpoints.save(
            task_id=task.task_id,
            step=task.step,
            task_state=self._task_state(task),
            model_state=task.model_state,
            backend=self.backend,
            trace_path=getattr(self.trace, "path", None),
        )
        self._record(
            "checkpoint_saved",
            task_id=task.task_id,
            step=task.step,
            state_size_bytes=getattr(metadata, "state_size_bytes", 0),
        )

    @staticmethod
    def _task_state(task: AgentTask) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "goal": task.goal,
            "workspace": str(task.workspace),
            "status": task.status.value,
            "step": task.step,
            "evidence_ids": list(task.evidence),
            "validation_failures": task.validation_failures,
            "tool_call_count": sum(isinstance(step.action, ToolCall) for step in task.steps),
            "steps": [AgentController._serialize_step(step) for step in task.steps],
        }

    @staticmethod
    def _serialize_step(step: AgentStep) -> dict[str, Any]:
        generation = None
        if step.generation is not None:
            generation = {
                # Stored for audit/debug only. Neural state is intentionally
                # never embedded here; it is persisted by the backend.
                "text": step.generation.text,
                "prompt_tokens": step.generation.prompt_tokens,
                "generated_tokens": step.generation.generated_tokens,
                "duration_ms": step.generation.duration_ms,
                "backend_name": step.generation.backend_name,
                "model_name": step.generation.model_name,
            }
        return {
            "number": step.number,
            "action": step.action.as_dict() if step.action else None,
            "observation": step.observation.as_dict() if step.observation else None,
            "generation": generation,
        }

    def _record(self, event: str, **fields: Any) -> None:
        if self.trace is not None:
            recorder = getattr(self.trace, "record", getattr(self.trace, "append", None))
            if recorder:
                recorder(event, **fields)
