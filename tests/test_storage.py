from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from statetrace.storage import TaskStore


class TaskStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = TaskStore(self.root / "tasks")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_round_trip_uses_a_file_below_the_store_root(self) -> None:
        path = self.store.save("task-safe_1.0", {"status": "COMPLETED"})
        self.assertEqual(path.parent, self.store.root)
        self.assertEqual(self.store.load("task-safe_1.0"), {"status": "COMPLETED"})

    def test_untrusted_task_ids_cannot_escape_or_address_other_paths(self) -> None:
        outside = self.root / "secret.json"
        outside.write_text('{"secret": true}', encoding="utf-8")

        for task_id in (
            "../secret",
            "task/child",
            "/absolute/path",
            "",
            ".hidden",
            "task name",
            "a" * 129,
        ):
            with self.subTest(task_id=task_id):
                with self.assertRaises(ValueError):
                    self.store.load(task_id)
                with self.assertRaises(ValueError):
                    self.store.save(task_id, {"unexpected": True})

        self.assertEqual(outside.read_text(encoding="utf-8"), '{"secret": true}')


if __name__ == "__main__":
    unittest.main()

