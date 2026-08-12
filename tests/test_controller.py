from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from statetrace.backends.base import GenerationResult, SamplingConfig
from statetrace.backends.replay import ReplayBackend
from statetrace.checkpoints import CheckpointManager
from statetrace.controller import AgentController, ControllerConfig
from statetrace.models import AgentStatus, ErrorCode
from statetrace.tools import default_registry


class RejectOnceValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, report, evidence):
        self.calls += 1
        passed = self.calls > 1
        return type(
            "Result",
            (),
            {
                "passed": passed,
                "as_observation": lambda _: {
                    "status": "validation_failed",
                    "errors": [{"code": "MISSING_EVIDENCE"}],
                },
            },
        )()


class RecordingStateBackend:
    """Tiny stateful backend that exposes prompts for recurrence tests."""

    name = "recording_state"
    model_name = "test-rwkv"
    context_mode = "incremental_state"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.states: list[Any] = []

    def generate(
        self,
        prompt: str,
        state: Any | None,
        sampling: SamplingConfig,
    ) -> GenerationResult:
        self.prompts.append(prompt)
        self.states.append(state)
        if state is None:
            action = (
                '{"type":"tool_call","thought_summary":"read source",'
                '"tool":"read_file","arguments":{"path":"app.py","start_line":1,"end_line":2}}'
            )
            return GenerationResult(text=action, state={"native": "state-after-step-1"})
        import json

        return GenerationResult(text=json.dumps(final_action()), state={"native": "state-after-step-2"})


def final_action(evidence_id: str = "obs-001") -> dict:
    return {
        "type": "final",
        "tool": "finish_report",
        "arguments": {
            "summary": "The return value is visible.",
            "findings": [
                {
                    "file": "app.py",
                    "line": 2,
                    "claim": "It returns 41.",
                    "evidence_ids": [evidence_id],
                }
            ],
            "verification": {},
            "recommendations": ["Return the intended value."],
        },
    }


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "app.py").write_text("def value():\n    return 41\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def controller(self, responses, **kwargs):
        return AgentController(
            backend=ReplayBackend(responses),
            tools=default_registry(),
            config=ControllerConfig(max_steps=len(responses)),
            **kwargs,
        )

    def test_strict_loop_collects_evidence_then_finishes(self):
        responses = [
            {
                "type": "tool_call",
                "thought_summary": "read source",
                "tool": "read_file",
                "arguments": {"path": "app.py", "start_line": 1, "end_line": 2},
            },
            final_action(),
        ]
        controller = self.controller(responses)
        result = controller.run(controller.new_task("inspect", self.root, "task-ok"))
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertIn("obs-001", result.evidence)

    def test_protocol_error_is_observed_and_next_response_can_recover(self):
        responses = ["not json", final_action("missing-but-unvalidated")]
        controller = self.controller(responses)
        result = controller.run(controller.new_task("recover", self.root, "task-protocol"))
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.steps[0].action.tool, "__protocol__")
        self.assertEqual(result.steps[0].observation.error_code, ErrorCode.INVALID_MODEL_OUTPUT)

    def test_validator_rejection_returns_to_model(self):
        validator = RejectOnceValidator()
        responses = [final_action(), final_action()]
        controller = self.controller(responses, validator=validator)
        result = controller.run(controller.new_task("validate", self.root, "task-validation"))
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.validation_failures, 1)
        self.assertEqual(result.steps[0].observation.status, "validation_failed")

    def test_duplicate_tool_call_is_blocked_after_limit(self):
        call = {
            "type": "tool_call",
            "thought_summary": "repeat",
            "tool": "read_file",
            "arguments": {"path": "app.py", "start_line": 1, "end_line": 1},
        }
        controller = AgentController(
            backend=ReplayBackend([call, call, final_action()]),
            tools=default_registry(),
            config=ControllerConfig(max_steps=3, max_duplicate_calls=1),
        )
        result = controller.run(controller.new_task("dedupe", self.root, "task-repeat"))
        self.assertEqual(result.steps[1].observation.error_code, ErrorCode.DUPLICATE_TOOL_CALL)

    def test_checkpoint_restores_history_and_resumes_without_repeating_tools(self):
        call = {
            "type": "tool_call",
            "thought_summary": "read",
            "tool": "read_file",
            "arguments": {"path": "app.py", "start_line": 1, "end_line": 2},
        }
        checkpoints = CheckpointManager(self.root / ".state" / "checkpoints")
        first_backend = ReplayBackend([call])
        first = AgentController(
            backend=first_backend,
            tools=default_registry(),
            checkpoint_manager=checkpoints,
            config=ControllerConfig(max_steps=1),
        )
        interrupted = first.run(first.new_task("resume", self.root, "task-resume"))
        self.assertEqual(interrupted.status, AgentStatus.FAILED)
        self.assertEqual(checkpoints.list_steps("task-resume"), [0, 1])

        # Checkpoint step 1 predates the synthetic max-step terminal entry and
        # contains a cursor positioned after the already executed read_file.
        second_backend = ReplayBackend([call, final_action()])
        loaded = checkpoints.load(task_id="task-resume", backend=second_backend, step=1)
        second = AgentController(
            backend=second_backend,
            tools=default_registry(),
            checkpoint_manager=checkpoints,
            config=ControllerConfig(max_steps=2, max_duplicate_calls=1),
        )
        restored = second.restore_task(loaded)
        self.assertEqual(len(restored.steps), 1)
        self.assertIn("obs-001", restored.evidence)
        completed = second.run(restored)
        self.assertEqual(completed.status, AgentStatus.COMPLETED)
        self.assertEqual(len([s for s in completed.steps if getattr(s.action, "tool", None) == "read_file"]), 1)

    def test_recurrent_state_uses_incremental_observation_after_first_turn(self):
        backend = RecordingStateBackend()
        controller = AgentController(
            backend=backend,
            tools=default_registry(),
            config=ControllerConfig(max_steps=2),
        )
        result = controller.run(controller.new_task("inspect unique-goal-marker", self.root, "task-incremental"))

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("unique-goal-marker", backend.prompts[0])
        self.assertNotIn("unique-goal-marker", backend.prompts[1])
        self.assertIn("Environment observation", backend.prompts[1])
        self.assertEqual(backend.states[1], {"native": "state-after-step-1"})


if __name__ == "__main__":
    unittest.main()
