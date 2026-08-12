from __future__ import annotations

import io
import unittest
import urllib.error
from email.message import Message
from unittest.mock import patch

from statetrace.backends.api import RWKVAPIBackend
from statetrace.backends.base import SamplingConfig


class APIBackendTests(unittest.TestCase):
    def test_rejects_invalid_configuration_before_request(self) -> None:
        with self.assertRaises(ValueError):
            RWKVAPIBackend("localhost:8000", "rwkv")
        with self.assertRaises(ValueError):
            RWKVAPIBackend("https://example.test", "  ")
        with self.assertRaises(ValueError):
            RWKVAPIBackend("https://example.test", "rwkv", timeout_seconds=0)
        with self.assertRaises(ValueError):
            RWKVAPIBackend("https://user:secret@example.test/v1", "rwkv")

    def test_http_error_does_not_echo_sensitive_response_body(self) -> None:
        backend = RWKVAPIBackend("https://example.test/v1", "rwkv")
        error = urllib.error.HTTPError(
            backend.base_url,
            401,
            "unauthorized",
            Message(),
            io.BytesIO(b"api_key=super-secret prompt=private"),
        )
        with (
            patch("urllib.request.urlopen", side_effect=error),
            self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught,
        ):
            backend.generate("hello", None, SamplingConfig())
        self.assertNotIn("super-secret", str(caught.exception))

    def test_sampling_config_fails_fast(self) -> None:
        for arguments in (
            {"temperature": -0.1},
            {"temperature": "cold"},
            {"top_p": 0},
            {"top_p": 1.1},
            {"max_new_tokens": 0},
            {"max_new_tokens": "many"},
            {"seed": True},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SamplingConfig(**arguments)


if __name__ == "__main__":
    unittest.main()
