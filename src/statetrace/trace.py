"""Append-only, machine-readable execution traces.

The trace layer intentionally knows nothing about the agent controller.  Any
component can append an event and the resulting JSONL file remains readable
after an interrupted process (a partially written final line is ignored by
``read_events`` when requested).
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

_SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|password|secret|token)$", re.I)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with timezone information."""

    return datetime.now(UTC).isoformat()


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def redact(value: Any, *, workspace: Path | None = None) -> Any:
    """Recursively redact credentials and optionally shorten workspace paths.

    Redaction is deliberately conservative: any mapping key that looks like a
    credential is masked. Absolute paths below ``workspace`` become relative;
    other absolute paths are represented by their basename so reports do not
    disclose a user's home directory.
    """

    workspace = workspace.resolve() if workspace else None
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact(item, workspace=workspace)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact(item, workspace=workspace) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and os.path.isabs(value):
        path = Path(value)
        if workspace:
            try:
                return str(path.resolve().relative_to(workspace))
            except (OSError, ValueError):
                pass
        return f"<absolute-path>/{path.name}"
    return value


class TraceWriter:
    """Write durable JSONL events and compute a small metrics summary."""

    def __init__(self, path: str | Path, *, workspace: str | Path | None = None) -> None:
        self.path = Path(path)
        self.workspace = Path(workspace) if workspace is not None else None
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, /, **fields: Any) -> dict[str, Any]:
        if not event or not isinstance(event, str):
            raise ValueError("event must be a non-empty string")
        record = {"event": event, "timestamp": utc_now(), **fields}
        redacted = redact(record, workspace=self.workspace)
        if not isinstance(redacted, dict):
            raise TypeError("redacted trace record must remain a mapping")
        record = redacted
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    # ``record`` is a convenient alias for controller implementations.
    record = append

    def events(self, *, tolerate_partial: bool = True) -> list[dict[str, Any]]:
        return list(read_events(self.path, tolerate_partial=tolerate_partial))

    def summary(self) -> dict[str, Any]:
        return summarize_events(self.events())


def read_events(path: str | Path, *, tolerate_partial: bool = True) -> Iterator[dict[str, Any]]:
    """Yield JSONL events, optionally ignoring an interrupted final line."""

    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_partial and index == len(lines) - 1:
                continue
            raise
        if not isinstance(value, dict):
            raise ValueError(f"Trace line {index + 1} is not a JSON object")
        yield value


def summarize_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the stable metrics used by CLI status and reports."""

    result: dict[str, Any] = {
        "event_count": 0,
        "tool_call_count": 0,
        "tool_error_count": 0,
        "model_request_count": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "model_duration_ms": 0.0,
        "tool_duration_ms": 0.0,
        "validation_failure_count": 0,
        "latest_state_size_bytes": None,
        "final_status": None,
    }
    for item in events:
        result["event_count"] += 1
        kind = item.get("event")
        if kind == "tool_call":
            result["tool_call_count"] += 1
        elif kind == "tool_result":
            if item.get("status") == "error":
                result["tool_error_count"] += 1
            result["tool_duration_ms"] += float(item.get("duration_ms", 0) or 0)
        elif kind == "model_request":
            result["model_request_count"] += 1
            result["prompt_tokens"] += int(item.get("prompt_tokens", 0) or 0)
        elif kind == "model_response":
            result["generated_tokens"] += int(item.get("generated_tokens", 0) or 0)
            result["model_duration_ms"] += float(item.get("duration_ms", 0) or 0)
        elif kind in {"validation_failed", "validation_failure"}:
            result["validation_failure_count"] += 1
        elif kind == "checkpoint_saved":
            result["latest_state_size_bytes"] = item.get("state_size_bytes")
        elif kind in {"task_completed", "task_failed", "task_interrupted"}:
            result["final_status"] = kind.removeprefix("task_").upper()
    result["model_duration_ms"] = round(result["model_duration_ms"], 3)
    result["tool_duration_ms"] = round(result["tool_duration_ms"], 3)
    return result
