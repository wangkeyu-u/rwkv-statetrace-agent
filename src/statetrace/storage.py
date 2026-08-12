"""Small JSON task index used by the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TaskStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, task_id: str) -> Path:
        return self.root / f"{task_id}.json"

    def save(self, task_id: str, payload: dict[str, Any]) -> Path:
        path = self.path(task_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    def load(self, task_id: str) -> dict[str, Any]:
        path = self.path(task_id)
        if not path.exists():
            raise FileNotFoundError(f"Task not found: {task_id}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Task index is not a JSON object: {path}")
        return value
