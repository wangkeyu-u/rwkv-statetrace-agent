"""Minimal OpenAI-compatible text API backend for RWKV servers."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
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
        parsed_url = urllib.parse.urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
            raise ValueError("base_url must not contain credentials, a query, or a fragment")
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name.strip()
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
            # Response bodies may echo credentials or prompts. Keep the public
            # exception useful without carrying arbitrary server text into task
            # artifacts or user-visible reports.
            raise RuntimeError(f"RWKV API returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"RWKV API request failed: {type(exc).__name__}") from exc
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("RWKV API response does not contain assistant content") from exc
        if not isinstance(text, str):
            raise RuntimeError("RWKV API assistant content must be text")
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
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
