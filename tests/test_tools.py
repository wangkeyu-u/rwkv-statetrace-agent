from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from statetrace.models import ErrorCode
from statetrace.tools import default_registry
from statetrace.tools.base import ToolContext


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text(
            "def answer():\n    return 41\n", encoding="utf-8"
        )
        self.registry = default_registry()
        self.context = ToolContext(self.root, max_output_chars=2000)

    def tearDown(self):
        self.temp.cleanup()

    def test_list_read_and_search_are_bounded_and_structured(self):
        listing = self.registry.execute("list_files", {"path": "."}, self.context)
        self.assertEqual(listing.data["files"], ["src/app.py"])
        read = self.registry.execute(
            "read_file", {"path": "src/app.py", "start_line": 2, "end_line": 2}, self.context
        )
        self.assertIn("2 |", read.data["content"])
        search = self.registry.execute(
            "search_code", {"query": "return 41", "path": "."}, self.context
        )
        self.assertEqual(search.data["results"][0]["path"], "src/app.py")

    def test_path_escape_and_binary_are_rejected(self):
        escaped = self.registry.execute("read_file", {"path": "../outside"}, self.context)
        self.assertEqual(escaped.error_code, ErrorCode.PATH_OUTSIDE_WORKSPACE)
        (self.root / "bad.bin").write_bytes(b"hello\x00world")
        binary = self.registry.execute("read_file", {"path": "bad.bin"}, self.context)
        self.assertEqual(binary.error_code, ErrorCode.BINARY_FILE)

    def test_command_injection_is_rejected_before_execution(self):
        for command in (
            "pytest; echo pwned",
            "pytest && whoami",
            "$(whoami)",
            "`whoami`",
            "pytest ../../outside.py",
            "pytest --rootdir=/tmp",
            "pytest --junitxml=/tmp/results.xml",
            "pytest --cov-config=/tmp/coveragerc",
            "pytest --html=/tmp/report.html",
            "pytest -p no:cacheprovider",
            "npm test -- --runInBand",
        ):
            result = self.registry.execute("run_tests", {"command": command}, self.context)
            self.assertEqual(result.error_code, ErrorCode.COMMAND_NOT_ALLOWED, command)

    def test_bounded_pytest_selection_and_output_options_are_allowed(self):
        # Validate without executing pytest: the strict allow-list must retain
        # common selection and terminal-output flags.
        from statetrace.tools.tests import RunTestsTool

        for argv in (
            ["pytest", "-q", "-k", "date boundary", "--maxfail=2", "tests/test_dates.py"],
            ["python", "-m", "pytest", "-m", "not slow", "--tb", "short"],
            ["pytest", "--color=auto", "--disable-warnings", "src/test_app.py::test_value"],
        ):
            RunTestsTool._validate_arguments(argv, self.context)

    def test_calculator_never_evaluates_python(self):
        good = self.registry.execute("calculator", {"expression": "2 * (4 + 1)"}, self.context)
        self.assertEqual(good.data["result"], 10)
        bad = self.registry.execute(
            "calculator", {"expression": "__import__('os').getcwd()"}, self.context
        )
        self.assertEqual(bad.error_code, ErrorCode.INVALID_ARGUMENTS)

    def test_unknown_tool_is_model_visible_error(self):
        result = self.registry.execute("does_not_exist", {}, self.context)
        self.assertEqual(result.error_code, ErrorCode.UNKNOWN_TOOL)
        self.assertIn("allowed_tools", result.data)


if __name__ == "__main__":
    unittest.main()
