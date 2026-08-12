from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from statetrace.checkpoints import (
    CheckpointCorrupted,
    CheckpointManager,
    CheckpointNotFound,
    ModelStateMismatch,
)


class FakeBackend:
    name = "fake_rwkv"
    model_name = "tiny-test-model"
    state_kind = "rwkv_recurrent_tensors"
    supports_state = True

    def save_state(self, state, path: Path):
        payload = json.dumps(state, sort_keys=True).encode()
        path.write_bytes(payload)
        return {"dtype": "float32", "state_size_bytes": len(payload)}

    def load_state(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def clone_state(self, state):
        return copy.deepcopy(state)


class CheckpointManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manager = CheckpointManager(self.root / "checkpoints")
        self.backend = FakeBackend()

    def tearDown(self):
        self.temp.cleanup()

    def test_save_and_load_latest_checkpoint(self):
        trace = self.root / "trace.jsonl"
        trace.write_text('{"event":"tool_call"}\n', encoding="utf-8")
        metadata = self.manager.save(
            task_id="task-1",
            step=2,
            task_state={"goal": "diagnose", "status": "RUNNING"},
            model_state={"layers": [[1, 2]], "cursor": 3},
            backend=self.backend,
            trace_path=trace,
        )
        loaded = self.manager.load(task_id="task-1", backend=self.backend)
        self.assertEqual(loaded.task_state["task_id"], "task-1")
        self.assertEqual(loaded.task_state["step"], 2)
        self.assertEqual(loaded.model_state["cursor"], 3)
        self.assertGreater(metadata.state_size_bytes, 0)
        self.assertEqual(len(metadata.sha256 or ""), 64)
        self.assertIsNotNone(loaded.trace_path)

    def test_corrupted_state_fails_checksum(self):
        metadata = self.manager.save(
            task_id="task-corrupt",
            step=1,
            task_state={},
            model_state={"cursor": 1},
            backend=self.backend,
        )
        (metadata.path / "model_state.bin").write_bytes(b"tampered")
        with self.assertRaises(CheckpointCorrupted):
            self.manager.load(task_id="task-corrupt", backend=self.backend)

    def test_backend_or_model_mismatch_fails_closed(self):
        self.manager.save(
            task_id="task-mismatch",
            step=1,
            task_state={},
            model_state={"cursor": 1},
            backend=self.backend,
        )
        other = FakeBackend()
        other.model_name = "another-model"
        with self.assertRaises(ModelStateMismatch):
            self.manager.load(task_id="task-mismatch", backend=other)

    def test_clone_is_independent_and_updates_identity(self):
        self.manager.save(
            task_id="source",
            step=4,
            task_state={"history": ["one"]},
            model_state={"values": [1, 2]},
            backend=self.backend,
        )
        clone_path = self.manager.clone_checkpoint(
            source_task_id="source", new_task_id="branch", step=4
        )
        clone_task = json.loads((clone_path / "task_state.json").read_text(encoding="utf-8"))
        clone_task["history"].append("branch-only")
        (clone_path / "task_state.json").write_text(json.dumps(clone_task), encoding="utf-8")

        original = self.manager.load(task_id="source", step=4, backend=self.backend)
        branch = self.manager.load(task_id="branch", step=4, backend=self.backend)
        self.assertEqual(original.task_state["history"], ["one"])
        self.assertEqual(branch.task_state["task_id"], "branch")
        self.assertEqual(branch.task_state["forked_from"]["task_id"], "source")
        self.assertEqual(branch.model_state, original.model_state)

    def test_task_only_checkpoint_and_missing_task(self):
        no_state_backend = type(
            "NoState", (), {"name": "rwkv_api", "model_name": "api-model", "supports_state": False}
        )()
        self.manager.save(
            task_id="api-task", step=0, task_state={"status": "CREATED"}, backend=no_state_backend
        )
        loaded = self.manager.load(task_id="api-task", backend=no_state_backend)
        self.assertIsNone(loaded.model_state)
        self.assertEqual(loaded.metadata.state_size_bytes, 0)
        with self.assertRaises(CheckpointNotFound):
            self.manager.load(task_id="absent", backend=no_state_backend)

    def test_rejects_unsafe_task_id(self):
        with self.assertRaises(ValueError):
            self.manager.checkpoint_path("../escape", 1)


if __name__ == "__main__":
    unittest.main()
