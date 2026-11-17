"""Tests for the opencode V2 LLM backend (opencode_cli2.py).

Two classes:
- ``TestOpencodeLLM``: hermetic unit tests for the async adapter logic (the
  client's ``ask()`` is mocked, no server involved).
- ``TestOpencodeCli2Connectivity``: live connectivity tests against a running
  ``opencode serve``. The class is skipped automatically when the server is not
  reachable, so the default ``python -m unittest discover tests`` run stays
  green; with the server up, they run automatically.

Run (from the repo root, server on 127.0.0.1:4096)::

    python -m unittest discover tests -p "test_opencode_cli2_llm.py" -v
"""

import asyncio
import logging
import os
import unittest
from unittest.mock import MagicMock, patch

from openevolve.llm.opencode_cli2 import (
    Model,
    OpencodeClient,
    OpencodeLLM,
    TransportError,
    TurnResult,
    init_opencode_client,
)


def _make_cfg(**overrides):
    """Config stub with every attribute OpencodeLLM reads, defaulting to None."""
    defaults = {
        "name": None,
        "system_message": None,
        "api_base": None,
        "api_key": None,
        "opencode_api": None,
        "opencode_username": None,
        "timeout": None,
        "retries": None,
        "retry_delay": None,
        "opencode_agent": None,
        "opencode_allow_tools": None,
    }
    defaults.update(overrides)
    cfg = MagicMock()
    for key, value in defaults.items():
        setattr(cfg, key, value)
    return cfg


def _make_turn(text="ok", usage=None):
    return TurnResult(
        session_id="s1", message_id="m1", text=text, reasoning="",
        usage=usage or {}, cost=None, raw={},
    )


class TestOpencodeLLM(unittest.TestCase):
    """Hermetic tests: no server, the client's ask() is mocked."""

    @classmethod
    def setUpClass(cls):
        # Error-path tests emit expected ERROR/WARNING logs; silence them so
        # `python -m unittest` output stays clean.
        cls._oc_logger = logging.getLogger("opencode_client")
        cls._oc_logger_level = cls._oc_logger.level
        cls._oc_logger.setLevel(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        cls._oc_logger.setLevel(cls._oc_logger_level)

    def test_init_defaults(self):
        llm = OpencodeLLM(_make_cfg())
        self.assertEqual(llm.model, "deepseek/deepseek-flash")
        self.assertEqual(llm.api, "v2")
        self.assertEqual(llm.base_url, "http://127.0.0.1:4096")
        self.assertEqual(llm._model, Model("deepseek", "deepseek-flash"))
        self.assertEqual(llm.allow_tools, ())

    def test_init_overrides(self):
        llm = OpencodeLLM(_make_cfg(
            name="anthropic/claude-sonnet-4-5",
            api_base="http://localhost:9999",
            opencode_api="auto",
            timeout=30,
            retries=1,
            retry_delay=0,
        ))
        self.assertEqual(llm.model, "anthropic/claude-sonnet-4-5")
        self.assertEqual(llm._model, Model("anthropic", "claude-sonnet-4-5"))
        self.assertEqual(llm.base_url, "http://localhost:9999")
        self.assertEqual(llm.api, "auto")
        self.assertEqual(llm.timeout, 30)
        self.assertEqual(llm.retries, 1)

    def test_init_client_factory(self):
        self.assertIsInstance(init_opencode_client(_make_cfg()), OpencodeLLM)

    def test_generate_with_context(self):
        llm = OpencodeLLM(_make_cfg())
        fake = _make_turn("hello world", {"input": 10, "output": 5, "total": 15})
        with patch.object(OpencodeClient, "ask", return_value=fake) as ask:
            text = asyncio.run(llm.generate_with_context(
                "sys", [{"role": "user", "content": "hi"}]
            ))
        self.assertEqual(text, "hello world")
        kw = ask.call_args.kwargs
        self.assertEqual(kw["system"], "sys")
        self.assertTrue(kw["ephemeral"])
        self.assertEqual(kw["model"], llm._model)
        self.assertEqual(llm.last_usage["prompt_tokens"], 10)
        self.assertEqual(llm.last_usage["completion_tokens"], 5)
        self.assertEqual(llm.last_usage["total_tokens"], 15)

    def test_multiple_user_messages_joined(self):
        llm = OpencodeLLM(_make_cfg())
        captured = {}

        def fake_ask(prompt, **kw):
            captured["prompt"] = prompt
            return _make_turn()

        with patch.object(OpencodeClient, "ask", side_effect=fake_ask):
            asyncio.run(llm.generate_with_context(
                "sys",
                [
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "second"},
                    {"role": "assistant", "content": "ignored"},
                ],
            ))
        self.assertEqual(captured["prompt"], "first\n\nsecond")

    def test_retry_then_success(self):
        llm = OpencodeLLM(_make_cfg(retries=1, retry_delay=0))
        calls = {"n": 0}

        def fake_ask(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TransportError("boom")
            return _make_turn("ok")

        with patch.object(OpencodeClient, "ask", side_effect=fake_ask):
            text = asyncio.run(llm.generate_with_context("", [{"role": "user", "content": "x"}]))
        self.assertEqual(text, "ok")
        self.assertEqual(calls["n"], 2)

    def test_retry_exhausted(self):
        llm = OpencodeLLM(_make_cfg(retries=1, retry_delay=0))
        with patch.object(OpencodeClient, "ask", side_effect=TransportError("boom")):
            with self.assertRaises(TransportError):
                asyncio.run(llm.generate_with_context("", [{"role": "user", "content": "x"}]))

    def test_empty_response_raises(self):
        llm = OpencodeLLM(_make_cfg(retries=0))
        with patch.object(OpencodeClient, "ask", return_value=_make_turn("")):
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(llm.generate_with_context("", [{"role": "user", "content": "x"}]))
        self.assertIn("Empty response", str(ctx.exception))


BASE_URL = os.environ.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096")
USERNAME = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
PASSWORD = os.environ.get("OPENCODE_SERVER_PASSWORD", "p66qaOeW2G_lfZptHd_ehecZ5KdORQ0zq8-FeJy1oYo")
MODEL = os.environ.get("OPENCODE_MODEL", "deepseek/deepseek-flash")


class TestOpencodeCli2Connectivity(unittest.TestCase):
    """Live connectivity smoke tests.

    The whole class is skipped when `opencode serve` is not reachable, so the
    default test run stays green without a server; with the server running,
    these tests run automatically (no env var needed).
    """

    @classmethod
    def setUpClass(cls):
        probe = OpencodeClient(
            base_url=BASE_URL, api="auto", password=PASSWORD,
            username=USERNAME, timeout=5, connect_timeout=5,
        )
        try:
            probe.detect_api()
        except Exception as exc:
            raise unittest.SkipTest(
                f"`opencode serve` not reachable at {BASE_URL} ({exc})"
            )
        finally:
            probe.close()

    def _make_client(self, api="auto"):
        return OpencodeClient(
            base_url=BASE_URL, api=api, password=PASSWORD,
            username=USERNAME, timeout=120,
        )

    def test_01_detect_api(self):
        client = self._make_client()
        try:
            api = client.detect_api()
            self.assertEqual(api, "v2", f"expected v2, detected {api}")
        finally:
            client.close()

    def test_02_list_models(self):
        client = self._make_client()
        try:
            models = client.list_models()
            self.assertIsInstance(models, list)
            print("models:", [str(m) for m in models])
        finally:
            client.close()

    def test_03_client_ask_roundtrip(self):
        client = self._make_client()
        try:
            result = client.ask(
                "Reply with exactly: connectivity ok",
                system="You are a coding assistant. Answer in one short sentence.",
                ephemeral=True,
            )
            self.assertFalse(result.empty, "empty response from server")
            print("ask reply:", result.text.strip())
            print("ask usage:", result.usage)
        finally:
            client.close()

    def test_04_opencode_llm_roundtrip(self):
        llm = OpencodeLLM(_make_cfg(
            name=MODEL,
            api_base=BASE_URL,
            api_key=PASSWORD,
            opencode_api="auto",
            opencode_username=USERNAME,
            timeout=120,
            retries=0,
            retry_delay=5,
        ))
        try:
            text = asyncio.run(llm.generate_with_context(
                "You are a coding assistant. Answer in one short sentence.",
                [{"role": "user", "content": "Reply with exactly: llm backend ok"}],
            ))
            self.assertTrue(text.strip(), "empty LLM response")
            print("llm reply:", text.strip())
            print("negotiated api:", llm.client.api)
            print("llm last_usage:", llm.last_usage)
        finally:
            llm.client.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
