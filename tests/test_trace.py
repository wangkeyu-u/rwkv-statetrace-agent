from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from statetrace.trace import TraceWriter, read_events, redact, summarize_events


class TraceTests(unittest.TestCase):
    def test_append_read_redact_and_summarize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = TraceWriter(root / "run.jsonl", workspace=root)
            first = trace.append(
                "model_request", prompt_tokens=12, api_key="do-not-store", path=root / "src.py"
            )
            trace.append("model_response", generated_tokens=4, duration_ms=2.25)
            trace.append("tool_call", tool="read_file")
            trace.append("tool_result", status="error", duration_ms=1.5)
            trace.append("checkpoint_saved", state_size_bytes=99)
            trace.append("validation_failed")
            trace.append("task_completed")

            self.assertEqual(first["api_key"], "[REDACTED]")
            self.assertEqual(first["path"], "src.py")
            events = list(read_events(trace.path))
            self.assertEqual(len(events), 7)
            summary = summarize_events(events)
            self.assertEqual(summary["prompt_tokens"], 12)
            self.assertEqual(summary["generated_tokens"], 4)
            self.assertEqual(summary["tool_error_count"], 1)
            self.assertEqual(summary["latest_state_size_bytes"], 99)
            self.assertEqual(summary["final_status"], "COMPLETED")

    def test_partial_final_line_is_tolerated_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"event":"ok"}\n{"event":', encoding="utf-8")
            self.assertEqual([item["event"] for item in read_events(path)], ["ok"])
            with self.assertRaises(json.JSONDecodeError):
                list(read_events(path, tolerate_partial=False))

    def test_outside_absolute_path_is_not_disclosed(self):
        masked = redact("/Users/alice/private/secrets.txt", workspace=Path("/tmp/work"))
        self.assertEqual(masked, "<absolute-path>/secrets.txt")


if __name__ == "__main__":
    unittest.main()
