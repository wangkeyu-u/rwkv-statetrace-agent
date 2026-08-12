"""Deterministic recorded backend used by CI and the bundled demonstration."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .base import GenerationResult, SamplingConfig


class ReplayBackend:
    """Return recorded model decisions in order.

    Replay is deliberately labelled as non-live inference. Its small cursor is
    checkpointable only to make controller recovery testable; it is not an
    RWKV neural state and must never be presented as one.
    """

    name = "replay"
    model_name = "recorded-demo"
    supports_state = True
    state_kind = "replay_cursor"
    is_live = False
    context_mode = "recorded"

    def __init__(self, responses: Iterable[str | dict[str, Any]]) -> None:
        self._responses = [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in responses
        ]

    @classmethod
    def from_file(cls, path: Path) -> ReplayBackend:
        data = json.loads(path.read_text(encoding="utf-8"))
        responses = data["responses"] if isinstance(data, dict) else data
        return cls(responses)

    def generate(
        self,
        prompt: str,
        state: dict[str, int] | None,
        sampling: SamplingConfig,
    ) -> GenerationResult:
        started = time.perf_counter()
        cursor = int((state or {}).get("cursor", 0))
        if cursor >= len(self._responses):
            raise RuntimeError("Replay responses exhausted before the task completed")
        text = self._responses[cursor]
        return GenerationResult(
            text=text,
            state={"cursor": cursor + 1},
            prompt_tokens=len(prompt.split()),
            generated_tokens=len(text.split()),
            duration_ms=(time.perf_counter() - started) * 1000,
            backend_name=self.name,
            model_name=self.model_name,
        )

    def save_state(self, state: Any, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, sort_keys=True).encode()
        path.write_bytes(payload)
        return {"state_kind": self.state_kind, "state_size_bytes": len(payload)}

    def load_state(self, path: Path) -> dict[str, int]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("cursor"), int):
            raise ValueError(f"Invalid replay state: {path}")
        return {"cursor": value["cursor"]}

    def clone_state(self, state: Any) -> Any:
        return copy.deepcopy(state)
