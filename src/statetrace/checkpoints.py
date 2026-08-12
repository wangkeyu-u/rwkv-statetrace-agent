"""Crash-resistant task and model-state checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

from .trace import utc_now

FORMAT_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointNotFound(CheckpointError):
    pass


class CheckpointCorrupted(CheckpointError):
    pass


class ModelStateMismatch(CheckpointError):
    pass


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    task_id: str
    step: int
    path: Path
    backend: str
    model_name: str
    state_kind: str | None
    state_size_bytes: int
    sha256: str | None
    created_at: str


@dataclass(slots=True)
class LoadedCheckpoint:
    task_state: dict[str, Any]
    model_state: Any
    metadata: CheckpointMetadata
    trace_path: Path | None = None


def _serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError("task_state must be a mapping, dataclass, or serializable model")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_integrity_manifest(directory: Path) -> None:
    """Hash every durable checkpoint artifact except the manifest itself."""

    entries = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == "integrity.sha256":
            continue
        if path.is_symlink() or not path.is_file():
            raise CheckpointError(f"Unexpected checkpoint artifact: {path.name}")
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "integrity.sha256").write_text("\n".join(entries) + "\n", encoding="ascii")


def _verify_integrity_manifest(directory: Path) -> None:
    manifest = directory / "integrity.sha256"
    if not manifest.is_file() or manifest.is_symlink():
        raise CheckpointCorrupted("Checkpoint integrity manifest is missing or invalid")
    expected: dict[str, str] = {}
    try:
        for line in manifest.read_text(encoding="ascii").splitlines():
            checksum, separator, name = line.partition("  ")
            if (
                not separator
                or not re.fullmatch(r"[0-9a-f]{64}", checksum)
                or not name
                or Path(name).name != name
                or name == manifest.name
                or name in expected
            ):
                raise ValueError("malformed integrity entry")
            expected[name] = checksum
    except (OSError, UnicodeError, ValueError) as exc:
        raise CheckpointCorrupted("Checkpoint integrity manifest is malformed") from exc
    try:
        actual_names = {
            path.name for path in directory.iterdir() if path.name != manifest.name
        }
        if actual_names != set(expected):
            raise CheckpointCorrupted("Checkpoint artifact set does not match integrity manifest")
        for name, checksum in expected.items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or _sha256(path) != checksum:
                raise CheckpointCorrupted(f"Checkpoint artifact checksum mismatch: {name}")
    except CheckpointCorrupted:
        raise
    except OSError as exc:
        raise CheckpointCorrupted("Checkpoint artifacts could not be read safely") from exc


def _validate_id(value: str, label: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} contains unsafe characters: {value!r}")
    return value


class CheckpointManager:
    """Save, verify, load, list, and clone task checkpoints.

    Backends own their state serialization. This avoids converting native RWKV
    tensors through a lossy generic format. A backend without state support may
    still save task state; its metadata explicitly records no model state.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, task_id: str, step: int) -> Path:
        _validate_id(task_id, "task_id")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")
        return self.root / task_id / f"step_{step:04d}"

    def save(
        self,
        *,
        task_id: str,
        step: int,
        task_state: Any,
        model_state: Any = None,
        backend: Any = None,
        trace_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> CheckpointMetadata:
        destination = self.checkpoint_path(task_id, step)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        created_at = utc_now()
        try:
            state_payload = _serializable(task_state)
            state_payload.setdefault("task_id", task_id)
            state_payload.setdefault("step", step)
            (temp / "task_state.json").write_text(
                json.dumps(state_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )

            backend_name = str(getattr(backend, "name", "none"))
            model_name = str(getattr(backend, "model_name", "none"))
            state_kind = getattr(backend, "state_kind", None)
            state_size = 0
            checksum: str | None = None
            backend_metadata: dict[str, Any] = {}
            state_file = temp / "model_state.bin"
            supports_state = bool(getattr(backend, "supports_state", False))
            if model_state is not None:
                if backend is None or not supports_state:
                    raise CheckpointError("A model state was supplied but the backend cannot persist it")
                backend_metadata = dict(backend.save_state(model_state, state_file) or {})
                if not state_file.is_file():
                    raise CheckpointError("Backend save_state did not create model_state.bin")
                state_size = state_file.stat().st_size
                checksum = _sha256(state_file)
                (temp / "checksum.sha256").write_text(
                    f"{checksum}  model_state.bin\n", encoding="ascii"
                )

            meta = {
                "format_version": FORMAT_VERSION,
                "task_id": task_id,
                "step": step,
                "backend": backend_name,
                "model_name": model_name,
                "state_kind": state_kind,
                "has_model_state": model_state is not None,
                "state_size_bytes": state_size,
                "sha256": checksum,
                "created_at": created_at,
                "backend_metadata": backend_metadata,
            }
            (temp / "model_state.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            if trace_path is not None:
                source = Path(trace_path)
                if source.exists():
                    shutil.copy2(source, temp / "trace.jsonl")

            _write_integrity_manifest(temp)

            if destination.exists():
                if not overwrite:
                    raise FileExistsError(destination)
                shutil.rmtree(destination)
            os.replace(temp, destination)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        return CheckpointMetadata(
            task_id=task_id,
            step=step,
            path=destination,
            backend=backend_name,
            model_name=model_name,
            state_kind=state_kind,
            state_size_bytes=state_size,
            sha256=checksum,
            created_at=created_at,
        )

    def list_steps(self, task_id: str) -> list[int]:
        _validate_id(task_id, "task_id")
        task_dir = self.root / task_id
        if not task_dir.exists():
            return []
        steps: list[int] = []
        for path in task_dir.iterdir():
            match = re.fullmatch(r"step_(\d{4,})", path.name)
            if path.is_dir() and match:
                steps.append(int(match.group(1)))
        return sorted(steps)

    def latest_step(self, task_id: str) -> int:
        steps = self.list_steps(task_id)
        if not steps:
            raise CheckpointNotFound(f"No checkpoints found for task {task_id!r}")
        return steps[-1]

    def load(self, *, task_id: str, backend: Any, step: int | None = None) -> LoadedCheckpoint:
        selected_step = self.latest_step(task_id) if step is None else step
        path = self.checkpoint_path(task_id, selected_step)
        if not path.is_dir():
            raise CheckpointNotFound(f"Checkpoint does not exist: {path}")
        _verify_integrity_manifest(path)
        try:
            task_state = json.loads((path / "task_state.json").read_text(encoding="utf-8"))
            raw_meta = json.loads((path / "model_state.meta.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise CheckpointCorrupted(f"Checkpoint metadata is incomplete or invalid: {path}") from exc

        if raw_meta.get("format_version") != FORMAT_VERSION:
            raise CheckpointCorrupted(
                f"Unsupported checkpoint format {raw_meta.get('format_version')!r}"
            )
        if raw_meta.get("task_id") != task_id or raw_meta.get("step") != selected_step:
            raise CheckpointCorrupted("Checkpoint metadata identity does not match its storage path")
        if task_state.get("task_id") != task_id or task_state.get("step") != selected_step:
            raise CheckpointCorrupted("Task state identity does not match its storage path")
        expected_backend = str(getattr(backend, "name", "none"))
        expected_model = str(getattr(backend, "model_name", "none"))
        if raw_meta.get("backend") != expected_backend:
            raise ModelStateMismatch(
                f"Checkpoint backend {raw_meta.get('backend')!r} does not match {expected_backend!r}"
            )
        if raw_meta.get("model_name") != expected_model:
            raise ModelStateMismatch(
                f"Checkpoint model {raw_meta.get('model_name')!r} does not match {expected_model!r}"
            )

        model_state = None
        if raw_meta.get("has_model_state"):
            state_file = path / "model_state.bin"
            if not state_file.is_file():
                raise CheckpointCorrupted("Checkpoint declares model state but the state file is missing")
            expected_hash = raw_meta.get("sha256")
            actual_hash = _sha256(state_file)
            if not expected_hash or actual_hash != expected_hash:
                raise CheckpointCorrupted(
                    f"Model state checksum mismatch: expected {expected_hash}, got {actual_hash}"
                )
            if raw_meta.get("state_size_bytes") != state_file.stat().st_size:
                raise CheckpointCorrupted("Model state size does not match checkpoint metadata")
            try:
                model_state = backend.load_state(state_file)
            except Exception as exc:
                raise CheckpointCorrupted("Backend could not load the saved model state") from exc

        metadata = CheckpointMetadata(
            task_id=str(raw_meta["task_id"]),
            step=int(raw_meta["step"]),
            path=path,
            backend=str(raw_meta["backend"]),
            model_name=str(raw_meta["model_name"]),
            state_kind=raw_meta.get("state_kind"),
            state_size_bytes=int(raw_meta.get("state_size_bytes", 0)),
            sha256=raw_meta.get("sha256"),
            created_at=str(raw_meta["created_at"]),
        )
        trace_path = path / "trace.jsonl"
        return LoadedCheckpoint(
            task_state=task_state,
            model_state=model_state,
            metadata=metadata,
            trace_path=trace_path if trace_path.exists() else None,
        )

    def clone_checkpoint(
        self,
        *,
        source_task_id: str,
        new_task_id: str,
        step: int | None = None,
    ) -> Path:
        """Fork a checkpoint without mutating the source task."""

        _validate_id(new_task_id, "new_task_id")
        selected_step = self.latest_step(source_task_id) if step is None else step
        source = self.checkpoint_path(source_task_id, selected_step)
        if not source.is_dir():
            raise CheckpointNotFound(f"Checkpoint does not exist: {source}")
        # Never let a fork re-hash and thereby legitimize a damaged source.
        _verify_integrity_manifest(source)
        destination = self.checkpoint_path(new_task_id, selected_step)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        shutil.rmtree(temp)
        try:
            shutil.copytree(source, temp)
            state_path = temp / "task_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["task_id"] = new_task_id
            state["forked_from"] = {"task_id": source_task_id, "step": selected_step}
            state["updated_at"] = utc_now()
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            meta_path = temp / "model_state.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["task_id"] = new_task_id
            meta["forked_from"] = {"task_id": source_task_id, "step": selected_step}
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_integrity_manifest(temp)
            os.replace(temp, destination)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        return destination
