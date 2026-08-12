"""Optional adapter contract for native RWKV runtimes.

This module fails closed: a runtime must expose both generation and recurrent
state operations before it is advertised as a direct backend. Users can pass a
compatible adapter without coupling the core package to a large ML dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .base import GenerationResult, SamplingConfig


class NativeRWKVAdapter(Protocol):
    model_name: str

    def generate(self, prompt: str, state: Any, sampling: SamplingConfig) -> GenerationResult: ...
    def save_state(self, state: Any, path: Path) -> dict[str, Any]: ...
    def load_state(self, path: Path) -> Any: ...
    def clone_state(self, state: Any) -> Any: ...


class RWKVDirectBackend:
    name = "rwkv_direct"
    supports_state = True
    state_kind = "rwkv_recurrent_tensors"
    is_live = True
    # The recurrent state already contains the previous prompt and generated
    # action. Feeding the full transcript again would double-encode history.
    context_mode = "incremental_state"

    def __init__(self, adapter: NativeRWKVAdapter) -> None:
        missing = [name for name in ("generate", "save_state", "load_state", "clone_state") if not callable(getattr(adapter, name, None))]
        if missing:
            raise TypeError(f"Native RWKV adapter is missing state capabilities: {', '.join(missing)}")
        self.adapter = adapter
        self.model_name = adapter.model_name

    def generate(self, prompt: str, state: Any | None, sampling: SamplingConfig) -> GenerationResult:
        result = self.adapter.generate(prompt, state, sampling)
        result.backend_name = self.name
        result.model_name = self.model_name
        return result

    def save_state(self, state: Any, path: Path) -> dict[str, Any]:
        return self.adapter.save_state(state, path)

    def load_state(self, path: Path) -> Any:
        return self.adapter.load_state(path)

    def clone_state(self, state: Any) -> Any:
        return self.adapter.clone_state(state)
