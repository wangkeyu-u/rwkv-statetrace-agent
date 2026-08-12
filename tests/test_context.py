from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from statetrace.context import build_prompt
from statetrace.models import AgentStep, AgentTask, Observation, ToolCall


class ContextTests(unittest.TestCase):
    def test_bounded_history_remains_valid_json_and_keeps_latest_control_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = AgentTask("task-context", "inspect", Path(directory))
            for number in range(1, 4):
                task.steps.append(
                    AgentStep(
                        number,
                        ToolCall("read_file", {"path": f"file-{number}.py"}),
                        Observation(
                            status="success",
                            message="large",
                            data={"content": "x" * 5_000},
                            evidence_id=f"obs-{number:03d}",
                        ),
                    )
                )
            prompt = build_prompt(task, [], max_observation_chars=500)
            marker = "Recent execution state:\n"
            history_text = prompt.split(marker, 1)[1].split("\n\nReturn exactly", 1)[0]
            history = json.loads(history_text)
            self.assertLessEqual(len(history_text), 500)
            self.assertEqual(history[-1]["step"], 3)
            self.assertTrue(history[-1]["observation"]["prompt_truncated"])


if __name__ == "__main__":
    unittest.main()
