"""Backend-neutral model interfaces.

Only a direct backend that exposes recurrent tensors may claim native RWKV
state checkpoint support. Text APIs can implement generation while explicitly
reporting ``supports_state=False``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 0.2
    top_p: float = 0.8
    max_new_tokens: int = 256
    seed: int | None = 42


@dataclass(slots=True)
class GenerationResult:
    text: str
    state: Any = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    duration_ms: float = 0.0
    backend_name: str = "unknown"
    model_name: str = "unknown"


@runtime_checkable
class ModelBackend(Protocol):
    def generate(self, prompt: str, state: Any | None, sampling: SamplingConfig) -> GenerationResult: ...
    def save_state(self, state: Any, path: Path) -> Mapping[str, Any]: ...
    def load_state(self, path: Path) -> Any: ...
    def clone_state(self, state: Any) -> Any: ...


__all__ = ["GenerationResult", "ModelBackend", "SamplingConfig"]
