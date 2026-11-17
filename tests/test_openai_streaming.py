"""Tests for OpenAI chat-completion streaming."""

import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, Mock, patch

from openevolve.llm.openai import OpenAILLM


def _make_llm():
    model_cfg = Mock()
    model_cfg.name = "test-model"
    model_cfg.system_message = "system"
    model_cfg.temperature = 0.7
    model_cfg.top_p = None
    model_cfg.max_tokens = 100
    model_cfg.timeout = 60
    model_cfg.retries = 0
    model_cfg.retry_delay = 0
    model_cfg.api_base = "https://example.com/v1"
    model_cfg.api_key = None
    model_cfg.random_seed = None
    model_cfg.reasoning_effort = None
    with patch("openai.AsyncOpenAI"):
        return OpenAILLM(model_cfg)


class _AsyncChunks:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


def _chunk(content=None, usage=None):
    chunk = Mock()
    chunk.choices = [] if content is None else [Mock()]
    if content is not None:
        chunk.choices[0].delta.content = content
    chunk.usage = usage
    return chunk


class TestStreamingRequests(unittest.TestCase):
    def test_streams_and_concatenates_delta_content(self):
        llm = _make_llm()
        usage = Mock(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            completion_tokens_details=Mock(reasoning_tokens=2),
        )
        llm.client.chat.completions.create = AsyncMock(
            return_value=_AsyncChunks([_chunk("hel"), _chunk(None), _chunk("lo", usage)])
        )
        params = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}

        with self.assertLogs("openevolve.llm.openai", level=logging.INFO) as logs:
            self.assertEqual(asyncio.run(llm._call_api_stream(params)), "hello")
        llm.client.chat.completions.create.assert_awaited_once_with(
            **params, stream=True, stream_options={"include_usage": True}
        )
        self.assertNotIn("stream", params)
        self.assertEqual(llm.last_usage["total_tokens"], 5)
        self.assertEqual(llm.last_usage["reasoning_tokens"], 2)
        self.assertIn("reasoning_tokens=2", " ".join(logs.output))

    def test_preserves_existing_stream_options_and_enables_usage(self):
        llm = _make_llm()
        llm.client.chat.completions.create = AsyncMock(return_value=_AsyncChunks([]))
        params = {
            "model": "test-model",
            "messages": [],
            "stream_options": {"custom_option": True},
        }

        asyncio.run(llm._call_api_stream(params))

        llm.client.chat.completions.create.assert_awaited_once_with(
            model="test-model",
            messages=[],
            stream=True,
            stream_options={"custom_option": True, "include_usage": True},
        )
        self.assertEqual(params["stream_options"], {"custom_option": True})

    def test_requires_initialized_client(self):
        llm = _make_llm()
        llm.client = None

        with self.assertRaisesRegex(RuntimeError, "client is not initialized"):
            asyncio.run(llm._call_api_stream({"model": "test-model", "messages": []}))

    def test_adds_configured_reasoning_effort_without_mutating_params(self):
        llm = _make_llm()
        llm.reasoning_effort = "high"
        llm.client.chat.completions.create = AsyncMock(return_value=_AsyncChunks([]))
        params = {"model": "test-model", "messages": []}

        asyncio.run(llm._call_api_stream(params))

        llm.client.chat.completions.create.assert_awaited_once_with(
            model="test-model",
            messages=[],
            reasoning_effort="high",
            stream=True,
            stream_options={"include_usage": True},
        )
        self.assertNotIn("reasoning_effort", params)

    def test_explicit_reasoning_effort_overrides_model_config(self):
        llm = _make_llm()
        llm.reasoning_effort = "high"
        llm.client.chat.completions.create = AsyncMock(return_value=_AsyncChunks([]))
        params = {"model": "test-model", "messages": [], "reasoning_effort": "low"}

        asyncio.run(llm._call_api_stream(params))

        self.assertEqual(
            llm.client.chat.completions.create.await_args.kwargs["reasoning_effort"], "low"
        )

    def test_generate_with_context_uses_streaming_when_requested(self):
        llm = _make_llm()
        usage = Mock(prompt_tokens=4, completion_tokens=3, total_tokens=7)
        llm.client.chat.completions.create = AsyncMock(
            return_value=_AsyncChunks([_chunk("answer", usage)])
        )

        result = asyncio.run(
            llm.generate_with_context("system", [{"role": "user", "content": "question"}], stream=True)
        )

        self.assertEqual(result, "answer")
        self.assertEqual(llm.last_usage["total_tokens"], 7)
        self.assertTrue(llm.client.chat.completions.create.await_args.kwargs["stream"])


if __name__ == "__main__":
    unittest.main()
