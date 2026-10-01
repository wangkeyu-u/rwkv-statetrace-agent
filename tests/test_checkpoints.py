from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_failed_overwrite_publish_preserves_original_artifacts(self):
        trace = self.root / "trace.jsonl"
        trace.write_text('{"event":"original"}\n', encoding="utf-8")
        metadata = self.manager.save(
            task_id="task-publish", step=1, task_state={"goal": "original"},
            model_state={"cursor": 1}, backend=self.backend, trace_path=trace,
        )
        original = {path.name: path.read_bytes() for path in metadata.path.iterdir()}
        real_replace = os.replace

        def fail_publish(source, destination):
            if Path(source).name.startswith(".step_0001-"):
                raise OSError("injected publish failure")
            return real_replace(source, destination)

        with (
            patch("statetrace.checkpoints.os.replace", side_effect=fail_publish),
            self.assertRaisesRegex(OSError, "injected publish failure"),
        ):
            self.manager.save(
                task_id="task-publish", step=1, task_state={"goal": "replacement"},
                model_state={"cursor": 2}, backend=self.backend,
            )

        loaded = self.manager.load(task_id="task-publish", backend=self.backend)
        self.assertEqual(loaded.task_state["goal"], "original")
        self.assertEqual(loaded.model_state, {"cursor": 1})
        self.assertEqual(original, {path.name: path.read_bytes() for path in metadata.path.iterdir()})
        self.assertEqual(list(metadata.path.parent.iterdir()), [metadata.path])

    def test_interrupted_overwrite_recovers_previous_checkpoint(self):
        metadata = self.manager.save(
            task_id="task-interrupted", step=1, task_state={"goal": "original"},
            backend=None,
        )
        code = textwrap.dedent("""
            import os
            import sys
            from pathlib import Path
            from unittest.mock import patch
            from statetrace.checkpoints import CheckpointManager
            real_replace = os.replace
            def interrupt_publish(source, destination):
                if Path(source).name.startswith('.step_0001-'):
                    os._exit(91)
                return real_replace(source, destination)
            with patch('statetrace.checkpoints.os.replace', side_effect=interrupt_publish):
                CheckpointManager(sys.argv[1]).save(
                    task_id='task-interrupted', step=1, task_state={'goal': 'replacement'})
        """)
        result = subprocess.run([sys.executable, "-c", code, str(self.manager.root)], check=False)
        self.assertEqual(result.returncode, 91)

        recovered = CheckpointManager(self.manager.root)
        self.assertEqual(recovered.list_steps("task-interrupted"), [1])
        loaded = recovered.load(task_id="task-interrupted", backend=None)
        self.assertEqual(loaded.task_state["goal"], "original")
        self.assertIsNone(loaded.model_state)
        self.assertTrue(metadata.path.is_dir())

    def test_interrupted_backup_cleanup_keeps_published_replacement(self):
        self.manager.save(
            task_id="task-cleanup", step=1, task_state={"goal": "original"},
            model_state={"cursor": 1}, backend=self.backend,
        )
        real_rmtree = shutil.rmtree

        def interrupt_cleanup(path, *args, **kwargs):
            if Path(path).name.endswith(".backup"):
                raise KeyboardInterrupt("injected interruption after publish")
            return real_rmtree(path, *args, **kwargs)

        with (
            patch("statetrace.checkpoints.shutil.rmtree", side_effect=interrupt_cleanup),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.manager.save(
                task_id="task-cleanup", step=1, task_state={"goal": "replacement"},
                model_state={"cursor": 2}, backend=self.backend,
            )

        recovered = CheckpointManager(self.manager.root)
        loaded = recovered.load(task_id="task-cleanup", step=1, backend=self.backend)
        self.assertEqual(loaded.task_state["goal"], "replacement")
        self.assertEqual(loaded.model_state, {"cursor": 2})
        self.assertEqual(list(loaded.metadata.path.parent.iterdir()), [loaded.metadata.path])

    def test_recovery_does_not_hide_a_corrupted_published_checkpoint(self):
        self.manager.save(
            task_id="task-recovery-corrupt", step=0, task_state={"goal": "older"},
            model_state={"cursor": 0}, backend=self.backend,
        )
        metadata = self.manager.save(
            task_id="task-recovery-corrupt", step=1, task_state={"goal": "original"},
            model_state={"cursor": 1}, backend=self.backend,
        )
        backup = metadata.path.with_name(f".{metadata.path.name}.backup")
        shutil.copytree(metadata.path, backup)
        (metadata.path / "task_state.json").write_text('{"goal":"forged"}\n', encoding="utf-8")

        older = self.manager.load(task_id="task-recovery-corrupt", step=0, backend=self.backend)
        self.assertEqual(older.task_state["goal"], "older")
        with self.assertRaises(CheckpointCorrupted):
            self.manager.load(task_id="task-recovery-corrupt", backend=self.backend)
        self.assertTrue(backup.is_dir())

    def test_tampered_task_state_and_trace_fail_integrity(self):
        trace = self.root / "trace.jsonl"
        trace.write_text('{"event":"original"}\n', encoding="utf-8")
        metadata = self.manager.save(
            task_id="task-artifacts",
            step=1,
            task_state={"goal": "original"},
            model_state={"cursor": 1},
            backend=self.backend,
            trace_path=trace,
        )
        (metadata.path / "task_state.json").write_text('{"goal":"forged"}\n', encoding="utf-8")
        with self.assertRaises(CheckpointCorrupted):
            self.manager.load(task_id="task-artifacts", backend=self.backend)

        # Restore by saving atomically, then prove the trace is covered too.
        metadata = self.manager.save(
            task_id="task-artifacts",
            step=1,
            task_state={"goal": "original"},
            model_state={"cursor": 1},
            backend=self.backend,
            trace_path=trace,
        )
        (metadata.path / "trace.jsonl").write_text('{"event":"forged"}\n', encoding="utf-8")
        with self.assertRaises(CheckpointCorrupted):
            self.manager.load(task_id="task-artifacts", backend=self.backend)

    def test_unexpected_checkpoint_artifact_fails_integrity(self):
        metadata = self.manager.save(
            task_id="task-extra",
            step=1,
            task_state={},
            model_state={"cursor": 1},
            backend=self.backend,
        )
        (metadata.path / "injected.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaises(CheckpointCorrupted):
            self.manager.load(task_id="task-extra", backend=self.backend)

    def test_clone_refuses_to_bless_a_corrupted_source(self):
        metadata = self.manager.save(
            task_id="damaged-source",
            step=3,
            task_state={"goal": "original"},
            model_state={"cursor": 3},
            backend=self.backend,
        )
        (metadata.path / "task_state.json").write_text('{"goal":"forged"}\n', encoding="utf-8")
        with self.assertRaises(CheckpointCorrupted):
            self.manager.clone_checkpoint(
                source_task_id="damaged-source", new_task_id="unsafe-branch", step=3
            )
        self.assertFalse((self.manager.root / "unsafe-branch").exists())

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
        branch = self.manager.load(task_id="branch", step=4, backend=self.backend)
        clone_task = json.loads((clone_path / "task_state.json").read_text(encoding="utf-8"))
        clone_task["history"].append("branch-only")
        (clone_path / "task_state.json").write_text(json.dumps(clone_task), encoding="utf-8")

        original = self.manager.load(task_id="source", step=4, backend=self.backend)
        self.assertEqual(original.task_state["history"], ["one"])
        self.assertEqual(branch.task_state["task_id"], "branch")
        self.assertEqual(branch.task_state["forked_from"]["task_id"], "source")
        self.assertEqual(branch.model_state, original.model_state)
        with self.assertRaises(CheckpointCorrupted):
            self.manager.load(task_id="branch", step=4, backend=self.backend)

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
