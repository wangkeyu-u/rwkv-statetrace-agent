from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from statetrace.models import AgentStep, FinalReport, Finding, Observation, ToolCall
from statetrace.validator import ReportValidator, evidence_from_steps


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "bug.py").write_text("def value():\n    return 41\n", encoding="utf-8")
        (self.workspace / "src" / "other.py").write_text("pass\n", encoding="utf-8")
        self.validator = ReportValidator(self.workspace)
        self.evidence = {
            "obs-1": {
                "id": "obs-1",
                "tool": "read_file",
                "arguments": {"path": "src/bug.py", "start_line": 1, "end_line": 2},
                "status": "success",
            },
            "obs-2": {
                "id": "obs-2",
                "tool": "run_tests",
                "arguments": {"command": "pytest -q"},
                "status": "success",
                "data": {
                    "exit_code": 1,
                    "stdout": "================ 1 failed, 7 passed in 0.12s ================",
                    "stderr": "",
                },
            },
        }

    def tearDown(self):
        self.temp.cleanup()

    def valid_report(self):
        return {
            "summary": "The function returns the wrong value.",
            "findings": [
                {
                    "file": "src/bug.py",
                    "line": 2,
                    "claim": "The implementation returns 41.",
                    "evidence_ids": ["obs-1"],
                }
            ],
            "verification": {"tests_run": ["pytest -q"], "result": "1 failed, 7 passed"},
            "recommendations": ["Return 42."],
        }

    def test_valid_report_passes(self):
        result = self.validator.validate(self.valid_report(), self.evidence)
        self.assertTrue(result.passed, result.errors)
        self.assertEqual(result.as_observation()["status"], "validation_passed")

    def test_accepts_final_report_model(self):
        report = FinalReport(
            summary="Wrong value.",
            findings=(Finding("src/bug.py", 2, "Returns 41.", ("obs-1",)),),
            verification={"tests_run": ["pytest -q"], "result": "tests failed"},
            recommendations=("Return 42.",),
        )
        self.assertTrue(self.validator.validate(report, self.evidence).passed)

    def test_rejects_missing_file_invalid_line_and_unknown_evidence(self):
        report = self.valid_report()
        report["findings"][0].update(
            {"file": "src/missing.py", "line": 99, "evidence_ids": ["obs-never"]}
        )
        codes = {error.code for error in self.validator.validate(report, self.evidence).errors}
        self.assertIn("FILE_NOT_FOUND", codes)
        self.assertIn("UNKNOWN_EVIDENCE", codes)

    def test_rejects_line_past_end_and_path_escape(self):
        report = self.valid_report()
        report["findings"].append(
            {"file": "../../etc/passwd", "line": 1, "claim": "escape", "evidence_ids": ["obs-1"]}
        )
        report["findings"][0]["line"] = 20
        codes = [error.code for error in self.validator.validate(report, self.evidence).errors]
        self.assertIn("INVALID_LINE_NUMBER", codes)
        self.assertIn("INVALID_FILE", codes)

    def test_rejects_evidence_for_a_different_file(self):
        report = self.valid_report()
        self.evidence["obs-1"]["arguments"]["path"] = "src/other.py"
        codes = {error.code for error in self.validator.validate(report, self.evidence).errors}
        self.assertIn("EVIDENCE_FILE_MISMATCH", codes)

    def test_rejects_unverified_test_claim(self):
        report = self.valid_report()
        report["verification"]["tests_run"] = ["python -m pytest"]
        codes = {error.code for error in self.validator.validate(report, self.evidence).errors}
        self.assertIn("UNVERIFIED_TEST_COMMAND", codes)

    def test_rejects_test_counts_that_contradict_captured_output(self):
        report = self.valid_report()
        report["verification"]["result"] = "8 passed"
        codes = {error.code for error in self.validator.validate(report, self.evidence).errors}
        self.assertIn("TEST_RESULT_MISMATCH", codes)

    def test_rejects_success_claim_when_test_exit_code_failed(self):
        report = self.valid_report()
        report["verification"]["result"] = "All tests passed successfully"
        codes = {error.code for error in self.validator.validate(report, self.evidence).errors}
        self.assertIn("TEST_RESULT_MISMATCH", codes)

    def test_ignores_failed_tool_attempt_as_test_evidence(self):
        report = self.valid_report()
        self.evidence["obs-2"]["status"] = "error"
        codes = {error.code for error in self.validator.validate(report, self.evidence).errors}
        self.assertIn("UNVERIFIED_TEST_RESULT", codes)

    def test_evidence_from_real_agent_steps(self):
        read_observation = Observation(status="success", evidence_id="obs-1")
        test_observation = Observation(
            status="success",
            evidence_id="obs-2",
            data={
                "exit_code": 1,
                "stdout": "1 failed, 7 passed in 0.12s",
                "stderr": "",
            },
        )
        steps = [
            AgentStep(1, ToolCall("read_file", {"path": "src/bug.py"}), read_observation),
            AgentStep(2, ToolCall("run_tests", {"command": "pytest -q"}), test_observation),
        ]
        joined = evidence_from_steps(steps)
        self.assertEqual(joined["obs-1"]["tool"], "read_file")
        self.assertTrue(self.validator.validate(self.valid_report(), joined).passed)


if __name__ == "__main__":
    unittest.main()
