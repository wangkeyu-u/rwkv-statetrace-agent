"""Minimal OpenAI-compatible text API backend for RWKV servers."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .base import GenerationResult, SamplingConfig


class RWKVAPIBackend:
    """Use an OpenAI-compatible RWKV endpoint.

    The common chat-completions protocol does not expose RWKV recurrent tensors,
    therefore this backend intentionally does not implement neural state
    checkpointing.
    """

    name = "rwkv_api"
    supports_state = False
    is_live = True
    context_mode = "full_transcript"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("RWKV_API_KEY", "")
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        state: Any | None,
        sampling: SamplingConfig,
    ) -> GenerationResult:
        if state is not None:
            raise ValueError("rwkv_api does not expose native recurrent state")
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_new_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"RWKV API returned HTTP {exc.code}: {detail}") from exc
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return GenerationResult(
            text=text,
            state=None,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            generated_tokens=int(usage.get("completion_tokens", 0)),
            duration_ms=(time.perf_counter() - started) * 1000,
            backend_name=self.name,
            model_name=self.model_name,
        )

    def save_state(self, state: Any, path: Path) -> dict[str, Any]:
        raise NotImplementedError("OpenAI-compatible API does not expose RWKV state")

    def load_state(self, path: Path) -> Any:
        raise NotImplementedError("OpenAI-compatible API does not expose RWKV state")

    def clone_state(self, state: Any) -> Any:
        raise NotImplementedError("OpenAI-compatible API does not expose RWKV state")
