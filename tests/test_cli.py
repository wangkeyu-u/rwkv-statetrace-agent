from __future__ import annotations

import argparse
import importlib.resources
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from statetrace.cli import _require_demo_dependencies, _state_root, _task_id


class CLITests(unittest.TestCase):
    def test_task_id_parser_matches_the_storage_safety_contract(self) -> None:
        self.assertEqual(_task_id("task-safe_1.0"), "task-safe_1.0")
        for value in ("../task", "/tmp/task", ".hidden", "task name"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                _task_id(value)

    def test_empty_statetrace_home_falls_back_to_local_directory(self) -> None:
        with patch.dict("os.environ", {"STATETRACE_HOME": ""}):
            self.assertEqual(_state_root(), Path(".statetrace").resolve())

    def test_demo_reports_the_install_command_when_pytest_is_missing(self) -> None:
        with (
            patch("statetrace.cli.util.find_spec", return_value=None),
            self.assertRaisesRegex(RuntimeError, r"\[demo\]"),
        ):
            _require_demo_dependencies()

    def test_packaged_demo_assets_match_the_inspectable_examples(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data = importlib.resources.files("statetrace").joinpath("demo_data")
        pairs = (
            ("replay_demo.json", project_root / "examples" / "replay_demo.json"),
            (
                "fixture/pyproject.toml",
                project_root / "examples" / "fixture_repo" / "pyproject.toml",
            ),
            (
                "fixture/src/calendar_edge/__init__.py",
                project_root / "examples" / "fixture_repo" / "src/calendar_edge/__init__.py",
            ),
            (
                "fixture/src/calendar_edge/dates.py",
                project_root / "examples" / "fixture_repo" / "src/calendar_edge/dates.py",
            ),
            (
                "fixture/tests/test_dates.py",
                project_root / "examples" / "fixture_repo" / "tests/test_dates.py",
            ),
        )
        for resource_name, example_path in pairs:
            with self.subTest(resource=resource_name):
                resource_text = data.joinpath(resource_name).read_text(encoding="utf-8")
                self.assertEqual(resource_text, example_path.read_text(encoding="utf-8"))

        replay = json.loads(data.joinpath("replay_demo.json").read_text(encoding="utf-8"))
        self.assertIn("not live model inference", replay["mode"])


if __name__ == "__main__":
    unittest.main()
