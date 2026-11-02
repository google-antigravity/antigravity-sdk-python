# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for LiteRTConnectionStrategy and LiteRTAgentConfig."""

import asyncio
import base64
import http.server
import json
import logging
import socketserver
import sys
from typing import Any
import unittest
from unittest import mock
import urllib.request


# 1. Inject mock litert_lm module to bypass ModuleNotFoundError
class MockEngine:

  def __init__(self, model_path, **kwargs):
    self.model_path = model_path
    self.kwargs = kwargs
    self.conversations = []

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

  def create_conversation(self, messages=None, tools=None, **kwargs):
    conv = MockConversation(messages, tools, **kwargs)
    self.conversations.append(conv)
    return conv


class MockConversation:

  def __init__(self, messages, tools=None, **kwargs):
    self.messages = messages or []
    self.tools = tools or []
    self.kwargs = kwargs
    self.simulated_response = None
    self.simulated_async_response = None
    self.exited = False
    self.cancelled = False
    self.sent_prompts = []

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True

  def cancel_process(self):
    self.cancelled = True

  # pylint: disable=unused-argument
  def send_message(self, prompt):
    self.sent_prompts.append(prompt)
    if self.simulated_response is not None:
      return self.simulated_response
    return {"content": [{"type": "text", "text": "Mocked sync reply."}]}

  # pylint: disable=unused-argument
  def send_message_async(self, prompt):
    self.sent_prompts.append(prompt)
    if self.simulated_async_response is not None:
      for chunk in self.simulated_async_response:
        yield chunk
      return
    yield {"content": [{"type": "text", "text": "Mocked "}]}
    yield {"content": [{"type": "text", "text": "async "}]}
    yield {"content": [{"type": "text", "text": "reply."}]}


# Define mock structures to match litert_messages parser expectations
class MockText:

  def __init__(self, text):
    self.text = text

  def to_json(self):
    return {"type": "text", "text": self.text}

  def __eq__(self, other):
    return isinstance(other, MockText) and self.text == other.text


class MockToolResponse:

  def __init__(self, name, response):
    self.name = name
    self.response = response

  def to_json(self):
    return {
        "type": "tool_response",
        "name": self.name,
        "response": self.response,
    }

  def __eq__(self, other):
    return (
        isinstance(other, MockToolResponse)
        and self.name == other.name
        and self.response == other.response
    )


class MockContents:

  def __init__(self, contents):
    self.contents = contents

  @classmethod
  def of(cls, *args):
    if not args:
      return cls([])
    if len(args) == 1:
      arg = args[0]
      if isinstance(arg, str):
        return cls([MockText(arg)])
      if isinstance(arg, list):
        return cls(list(arg))
      return cls([arg])
    items = []
    for arg in args:
      if isinstance(arg, str):
        items.append(MockText(arg))
      elif isinstance(arg, list):
        items.extend(arg)
      else:
        items.append(arg)
    return cls(items)

  @classmethod
  def empty(cls):
    return cls([])

  def __eq__(self, other):
    return isinstance(other, MockContents) and self.contents == other.contents


class MockToolCall:

  def __init__(self, name: str, arguments: dict[str, Any]):
    self.name = name
    self.arguments = arguments

  def to_json(self):
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "arguments": self.arguments,
        },
    }

  def __eq__(self, other):
    return (
        isinstance(other, MockToolCall)
        and self.name == other.name
        and self.arguments == other.arguments
    )


class MockTool:

  def get_tool_description(self) -> dict[str, Any]:
    return {}

  def execute(self, param: Any) -> Any:
    del param  # Unused
    return None


class MockImageBytes:

  def __init__(self, data: bytes):
    self.bytes = data

  def to_json(self):
    return {
        "type": "image",
        "blob": base64.b64encode(self.bytes).decode("utf-8"),
    }

  def __eq__(self, other):
    return isinstance(other, MockImageBytes) and self.bytes == other.bytes


class MockImageFile:

  def __init__(self, path: str):
    self.absolute_path = path

  def to_json(self):
    return {"type": "image", "path": self.absolute_path}

  def __eq__(self, other):
    return (
        isinstance(other, MockImageFile)
        and self.absolute_path == other.absolute_path
    )


class MockAudioBytes:

  def __init__(self, data: bytes):
    self.bytes = data

  def to_json(self):
    return {
        "type": "audio",
        "blob": base64.b64encode(self.bytes).decode("utf-8"),
    }

  def __eq__(self, other):
    return isinstance(other, MockAudioBytes) and self.bytes == other.bytes


class MockAudioFile:

  def __init__(self, path: str):
    self.absolute_path = path

  def to_json(self):
    return {"type": "audio", "path": self.absolute_path}

  def __eq__(self, other):
    return (
        isinstance(other, MockAudioFile)
        and self.absolute_path == other.absolute_path
    )


class MockSupportedModalities:

  def __init__(
      self, vision: bool = False, audio: bool = False, text: bool = True
  ):
    self.vision = vision
    self.audio = audio
    self.text = text


class MockModelInfo:

  def __init__(self, model_path: str):
    self.model_path = model_path
    self.input_modalities = MockSupportedModalities()
    self.closed = False

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def supported_backends_for_modality(self, modality):
    del modality  # Unused.
    return ["cpu"]

  def close(self):
    self.closed = True


mock_litert = mock.MagicMock()
mock_litert.Engine = MockEngine
mock_litert.Backend.CPU = lambda: "cpu"
mock_litert.Backend.GPU = lambda: "gpu"
mock_litert.Backend.NPU = lambda: "npu"
mock_litert.Message.system = lambda t: {"role": "system", "content": t}
mock_litert.Message.user = lambda t: {"role": "user", "content": t}
mock_litert.Message.model = lambda contents=None, tool_calls=None, **kw: {
    "role": "assistant",
    "content": contents,
    "tool_calls": tool_calls,
}
mock_litert.Message.tool = lambda c: {"role": "tool", "content": c}
mock_litert.Contents = MockContents
mock_litert.Content.Text = MockText
mock_litert.Content.ImageBytes = MockImageBytes
mock_litert.Content.ImageFile = MockImageFile
mock_litert.Content.AudioBytes = MockAudioBytes
mock_litert.Content.AudioFile = MockAudioFile
mock_litert.Content.ToolResponse = MockToolResponse
mock_litert.Tool = MockTool
mock_litert.ToolCall = MockToolCall
mock_litert.ModelInfo = MockModelInfo
mock_litert.LiteRtLmModality = mock.MagicMock()
mock_litert.LiteRtLmModality.VISION = "vision"

sys.modules["litert_lm"] = mock_litert

# pylint: disable=g-import-not-at-top
from google.antigravity.proto import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local import litert_connection
from google.antigravity.connections.local import litert_connection_config
from google.antigravity.connections.local import litert_server
from google.antigravity.connections.local import local_connection
from google.antigravity.connections.local import test_utils
from google.antigravity.hooks import policy
from google.antigravity.tools import tool_runner as tool_runner_mod

# pylint: enable=g-import-not-at-top


_urlopen_no_proxy = litert_connection._urlopen_no_proxy


class LiteRTConnectionTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_litert_strategy_lifecycle_and_server(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify LiteRT loopback server binds cleanly, responds to health and POST stream."""
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,  # Bypass GPU check
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )

    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    # 1. Health Check GET /v1/models
    url = f"{strategy._openai_server_url}/v1/models"
    with _urlopen_no_proxy(url) as r:
      self.assertEqual(r.status, 200)
      data = json.loads(r.read().decode("utf-8"))
      self.assertEqual(data["data"][0]["id"], "model.litertlm")

    # 2. Synchronous Completions POST
    payload = {
        "model": "gemma",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req) as r:
      self.assertEqual(r.status, 200)
      data = json.loads(r.read().decode("utf-8"))
      self.assertEqual(
          data["choices"][0]["message"]["content"], "Mocked sync reply."
      )

    # 3. Streaming Completions POST (SSE verification)
    payload["stream"] = True
    req_stream = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req_stream) as r:
      self.assertEqual(r.status, 200)
      lines = []
      while True:
        line = r.readline().decode("utf-8")
        if not line:
          break
        line = line.strip()
        if line:
          lines.append(line)
          if "[DONE]" in line:
            break

      # Extract non-empty SSE blocks
      events = [line for line in lines if line.startswith("data: ")]
      self.assertGreaterEqual(len(events), 4)

      # Parse first chunk containing assistant role
      first_chunk = json.loads(events[0][6:])
      self.assertEqual(first_chunk["choices"][0]["delta"]["role"], "assistant")

      # Parse subsequent text delta chunks
      text_delta = "".join(
          json.loads(event[6:])["choices"][0]["delta"].get("content", "")
          for event in events[1:]
          if "[DONE]" not in event
          and "delta" in json.loads(event[6:])["choices"][0]
      )
      self.assertEqual(text_delta, "Mocked async reply.")
      self.assertTrue(any("[DONE]" in line for line in lines))

    # 4. Multi-Role History Translation Verify
    payload_roles = {
        "model": "gemma",
        "messages": [
            {"role": "system", "content": "System directive"},
            {"role": "user", "content": "User text"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "tool", "name": "test_tool", "content": "Tool output"},
            {"role": "user", "content": "Follow up query"},
        ],
        "stream": False,
    }

    captured_messages_step4 = []
    original_create_conversation_step4 = (
        strategy._engine_context.create_conversation
    )

    def mock_create_conversation_step4(messages=None, tools=None, **kwargs):
      if messages:
        captured_messages_step4.extend(messages)
      return original_create_conversation_step4(messages, tools, **kwargs)

    strategy._engine_context.create_conversation = (
        mock_create_conversation_step4
    )

    req_roles = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload_roles).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req_roles) as r:
      self.assertEqual(r.status, 200)

    self.assertEqual(len(captured_messages_step4), 4)
    self.assertEqual(captured_messages_step4[0]["role"], "system")
    self.assertEqual(captured_messages_step4[1]["role"], "user")
    self.assertEqual(captured_messages_step4[2]["role"], "assistant")
    self.assertEqual(captured_messages_step4[3]["role"], "tool")

    strategy._engine_context.create_conversation = (
        original_create_conversation_step4
    )

    # 5. Malformed Payload 400 Bad Request Verify
    req_corrupted = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=b"{invalid_json}",
        headers={"Content-Type": "application/json"},
    )
    with self.assertRaises(urllib.error.HTTPError) as context:
      _urlopen_no_proxy(req_corrupted)
    self.assertEqual(context.exception.code, 400)

  @mock.patch("os.path.exists")
  def test_litert_harness_config(self, mock_exists):
    """Verify HarnessConfig clears Gemini and populates loopback URL settings."""
    mock_exists.return_value = True
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    strategy._openai_server_url = "http://127.0.0.1:54321"

    h_cfg = strategy._build_harness_config()
    self.assertEqual(len(h_cfg.models), 1)
    model = h_cfg.models[0]
    self.assertEqual(model.name, "model.litertlm")
    self.assertEqual(model.types, [localharness_pb2.MODEL_TYPE_TEXT])
    self.assertTrue(model.HasField("gemma_endpoint"))
    self.assertEqual(model.gemma_endpoint.base_url, "http://127.0.0.1:54321")

  @mock.patch("os.path.exists")
  def test_litert_config_compaction_config(self, mock_exists):
    mock_exists.return_value = True
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
        compaction_config=types.CompactionConfig(
            token_threshold=12345,
        ),
    )
    self.assertEqual(config.compaction_config.token_threshold, 12345)
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._max_kv_cache_tokens, 65536)
    self.assertEqual(strategy._compaction_config.token_threshold, 12345)

  @mock.patch("os.path.exists")
  def test_litert_config_tool_output_truncation_config(self, mock_exists):
    """Verify LiteRTAgentConfig forwards capabilities with tool_output_truncation_config to strategy and populates harness."""
    mock_exists.return_value = True
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
        capabilities=types.CapabilitiesConfig(
            tool_output_truncation_config=1024
        ),
    )
    self.assertEqual(
        config.capabilities.tool_output_truncation_config,
        types.ToolOutputTruncationConfig(max_tokens=1024),
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(
        strategy._capabilities_config.tool_output_truncation_config,
        types.ToolOutputTruncationConfig(max_tokens=1024),
    )
    strategy._openai_server_url = "http://127.0.0.1:54321"
    h_cfg = strategy._build_harness_config()
    self.assertTrue(h_cfg.HasField("tool_output_truncation"))
    self.assertTrue(h_cfg.tool_output_truncation.HasField("truncate"))
    self.assertEqual(h_cfg.tool_output_truncation.truncate.max_tokens, 1024)

  def test_litert_config_default_capabilities(self):
    """Verify LiteRTAgentConfig defaults to lightweight preset automatically."""
    litert_config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
    )
    self.assertEqual(
        litert_config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertEqual(
        litert_config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertFalse(litert_config.capabilities.enable_subagents)
    self.assertIsNotNone(litert_config.compaction_config)
    self.assertEqual(litert_config.compaction_config.token_threshold, 40960)

  def test_litert_config_lightweight_method(self):
    """Verify LiteRTAgentConfig.lightweight returns LiteRTAgentConfig with defaults."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    ).lightweight()
    self.assertIsInstance(config, litert_connection_config.LiteRTAgentConfig)
    self.assertEqual(config.model_path, "/tmp/model.litertlm")
    self.assertEqual(config.backend, litert_connection_config.LiteRTBackend.CPU)
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertIsNotNone(config.compaction_config)
    self.assertEqual(config.compaction_config.token_threshold, 40960)
    self.assertIsNone(config.capabilities.compaction_threshold)
    self.assertFalse(config.capabilities.enable_subagents)

    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._max_kv_cache_tokens, 65536)
    self.assertEqual(strategy._compaction_config.token_threshold, 40960)

  def test_litert_config_lightweight_method_with_overrides(self):
    """Verify LiteRTAgentConfig.lightweight respects capability overrides."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
        capabilities=types.CapabilitiesConfig(compaction_threshold=3000),
    ).lightweight()
    self.assertIsInstance(config, litert_connection_config.LiteRTAgentConfig)
    self.assertEqual(config.capabilities.compaction_threshold, 3000)
    self.assertIsNone(config.compaction_config)
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertFalse(config.capabilities.enable_subagents)

  def test_litert_config_constructor_with_capability_overrides(self):
    """Verify LiteRTAgentConfig constructor auto-applies lightweight preset with overrides."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
        capabilities=types.CapabilitiesConfig(compaction_threshold=3000),
    )
    self.assertIsInstance(config, litert_connection_config.LiteRTAgentConfig)
    self.assertEqual(config.capabilities.compaction_threshold, 3000)
    self.assertIsNone(config.compaction_config)
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertFalse(config.capabilities.enable_subagents)

  def test_litert_config_constructor_equals_lightweight_method(self):
    """Verify LiteRTAgentConfig(...) produces identical configuration to LiteRTAgentConfig(...).lightweight()."""
    config1 = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    config2 = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    ).lightweight()
    self.assertEqual(config1.capabilities, config2.capabilities)
    self.assertEqual(config1.compaction_config, config2.compaction_config)
    self.assertEqual(config1.model_dump(), config2.model_dump())

  def test_litert_config_lightweight_method_preserves_explicit_compaction_config(
      self,
  ):
    """Verify explicit compaction_config takes priority over lightweight preset."""
    custom_compaction = types.CompactionConfig(
        token_threshold=10000,
    )
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
        compaction_config=custom_compaction,
    ).lightweight()
    self.assertEqual(config.compaction_config, custom_compaction)
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(
        strategy._compaction_config.token_threshold, 10000
    )

  def test_litert_config_kv_cache_tokens_defaults(self):
    """Verify LiteRTAgentConfig defaults for KV cache tokens and derived compaction."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
    )
    self.assertIsNotNone(config.compaction_config)
    self.assertEqual(config.compaction_config.token_threshold, 40960)

    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._max_kv_cache_tokens, 65536)
    self.assertEqual(strategy._max_output_tokens, 16384)
    self.assertEqual(strategy._thinking_token_budget, 8192)
    self.assertEqual(strategy._compaction_config.token_threshold, 40960)

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_tool_call_id_backward_lookup(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )

    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    payload = {
        "model": "gemma",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "tc_123",
                    "type": "function",
                    "function": {
                        "name": "resolved_tool_name",
                        "arguments": "{}",
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "tc_123",
                "content": "Tool output",
            },
        ],
        "stream": False,
    }

    req = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req) as r:
      self.assertEqual(r.status, 200)

  def test_litert_config_validation_missing_model(self):

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/path/that/does/not/exist.litertlm",
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    with self.assertRaises(types.AntigravityValidationError):
      strategy._validate_connection()

  def test_format_openai_tool_call_helper(self):
    """Test _format_openai_tool_call helper handles dict and string args safely."""
    # Dict args -> JSON string
    tc1 = {"function": {"name": "my_tool", "arguments": {"arg1": "val1"}}}
    res1 = litert_server._format_openai_tool_call(tc1, 0, "12345")
    self.assertEqual(res1["function"]["name"], "my_tool")
    self.assertEqual(res1["function"]["arguments"], '{"arg1": "val1"}')

    # Pre-encoded string args -> preserved as string (no double encoding)
    tc2 = {"function": {"name": "my_tool", "arguments": '{"arg1": "val1"}'}}
    res2 = litert_server._format_openai_tool_call(tc2, 1, "12345")
    self.assertEqual(res2["function"]["arguments"], '{"arg1": "val1"}')

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_tools_payload_parsing(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )

    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    payload = {
        "model": "gemma",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "test_fn",
                "description": "A test function",
            },
        }],
        "stream": False,
    }

    req = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req) as r:
      self.assertEqual(r.status, 200)

  def test_litert_config_workspace_policies(self):
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        workspaces=["/tmp/my_workspace"],
    )
    # LiteRT uses localharness, which enforces workspace containment natively;
    # config.policies contains only the 2 confirm_run_command policies.
    self.assertEqual(len(config.policies), 2)
    self.assertEqual(config.policies[0].tool, "run_command")
    self.assertEqual(config.policies[1].tool, "*")

  @mock.patch(
      "google.antigravity.connections.local.litert_connection._check_gpu_acceleration_available"
  )
  def test_hardware_validation_gpu_warning(self, mock_gpu_check):
    """Verify hardware validation logs warning when GPU acceleration is missing."""
    mock_gpu_check.return_value = False
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.GPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    with self.assertLogs(level="WARNING") as cm:
      strategy._validate_hardware()
    self.assertTrue(
        any("GPU acceleration hardware" in log for log in cm.output)
    )

  @mock.patch(
      "google.antigravity.connections.local.litert_connection._check_gpu_acceleration_available"
  )
  def test_hardware_validation_gpu_success(self, mock_gpu_check):
    """Verify hardware validation succeeds silently when GPU is available."""
    mock_gpu_check.return_value = True
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.GPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    strategy._validate_hardware()

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_aenter_rollback_on_failure(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify __aenter__ error triggers cleanup of server and engine."""
    mock_exists.return_value = True
    mock_super_enter.side_effect = RuntimeError("Subprocess failed")
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    with (
        mock.patch.object(strategy, "_shutdown_server") as mock_shutdown,
        mock.patch.object(strategy, "_close_engine") as mock_close,
    ):
      with self.assertRaises(RuntimeError):
        await strategy.__aenter__()
      mock_shutdown.assert_called_once()
      mock_close.assert_called_once()

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_aenter_rollback_on_cancelled_error(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify __aenter__ BaseException cleans up server and engine."""
    mock_exists.return_value = True
    mock_super_enter.side_effect = asyncio.CancelledError()
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    with (
        mock.patch.object(
            strategy, "_shutdown_server", wraps=strategy._shutdown_server
        ) as mock_shutdown,
        mock.patch.object(
            strategy, "_close_engine", wraps=strategy._close_engine
        ) as mock_close,
    ):
      with self.assertRaises(asyncio.CancelledError):
        await strategy.__aenter__()
      mock_shutdown.assert_called_once()
      mock_close.assert_called_once()
      self.assertIsNone(strategy._openai_server)

  @mock.patch.object(litert_connection.litert_server, "_LITERT_AVAILABLE", True)
  def test_openai_tool_base_class_inheritance(self):
    """Verify OpenAITool inherits from litert_lm.Tool when litert_lm is available."""
    mock_tool_base = type("Tool", (object,), {})
    with mock.patch.object(
        litert_connection.litert_server, "litert_lm"
    ) as mock_lm:
      mock_lm.Tool = mock_tool_base
      # Re-evaluate class with mocked litert_lm.Tool
      tool = litert_connection.litert_server.OpenAITool({"name": "foo"})
      self.assertEqual(tool.get_tool_description(), {"name": "foo"})

  async def test_litert_engine_max_num_tokens_param(self):
    """Verify max_num_tokens is passed to litert_lm.Engine and decoupled from compaction."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/dummy/path.litertlm",
        compaction_config=types.CompactionConfig(
            token_threshold=21424,
        ),
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    mock_engine_cls = mock.MagicMock()
    mock_engine_inst = mock.MagicMock()
    mock_engine_cls.return_value = mock_engine_inst
    mock_engine_inst.__enter__.return_value = mock.MagicMock()

    with (
        mock.patch.object(litert_connection, "litert_lm") as mock_lm,
        mock.patch.object(
            litert_connection.litert_server, "LiteRTOpenAIServer"
        ),
        mock.patch.object(
            litert_connection, "_urlopen_no_proxy"
        ) as mock_urlopen,
        mock.patch("os.path.exists", return_value=True),
        mock.patch.object(
            local_connection.LocalConnectionStrategy,
            "__aenter__",
            return_value=None,
        ),
    ):
      mock_resp = mock.MagicMock()
      mock_resp.status = 200
      mock_resp.__enter__.return_value = mock_resp
      mock_urlopen.return_value = mock_resp
      mock_lm.Engine = mock_engine_cls
      mock_lm.Backend.CPU.return_value = "CPU"
      try:
        await strategy.__aenter__()
      finally:
        await strategy.__aexit__(None, None, None)

      mock_engine_cls.assert_called_once()
      _, kwargs = mock_engine_cls.call_args
      self.assertIn("max_num_tokens", kwargs)
      self.assertNotIn("max_context_tokens", kwargs)
      self.assertEqual(kwargs["max_num_tokens"], 65536)
      self.assertEqual(strategy._compaction_config.token_threshold, 21424)

  async def test_litert_engine_lightweight_decoupled_limits(self):
    """Verify lightweight preset passes max_kv_cache_tokens to Engine while deriving compaction."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/dummy/path.litertlm",
    ).lightweight()
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    mock_engine_cls = mock.MagicMock()
    mock_engine_inst = mock.MagicMock()
    mock_engine_cls.return_value = mock_engine_inst
    mock_engine_inst.__enter__.return_value = mock.MagicMock()

    with (
        mock.patch.object(litert_connection, "litert_lm") as mock_lm,
        mock.patch.object(
            litert_connection.litert_server, "LiteRTOpenAIServer"
        ),
        mock.patch.object(
            litert_connection, "_urlopen_no_proxy"
        ) as mock_urlopen,
        mock.patch("os.path.exists", return_value=True),
        mock.patch.object(
            local_connection.LocalConnectionStrategy,
            "__aenter__",
            return_value=None,
        ),
    ):
      mock_resp = mock.MagicMock()
      mock_resp.status = 200
      mock_resp.__enter__.return_value = mock_resp
      mock_urlopen.return_value = mock_resp
      mock_lm.Engine = mock_engine_cls
      mock_lm.Backend.CPU.return_value = "CPU"
      try:
        await strategy.__aenter__()
      finally:
        await strategy.__aexit__(None, None, None)

      mock_engine_cls.assert_called_once()
      _, kwargs = mock_engine_cls.call_args
      self.assertEqual(kwargs["max_num_tokens"], 65536)
      self.assertEqual(strategy._max_kv_cache_tokens, 65536)
      self.assertEqual(strategy._compaction_config.token_threshold, 40960)

  def test_litert_config_mcp_servers_and_subagents_passed_to_strategy(self):
    """Verify LiteRTAgentConfig passes mcp_servers and subagents to strategy."""
    mcp_server = types.McpStdioServer(
        name="test_mcp", command="echo", args=["hello"]
    )
    subagent = types.SubagentConfig(
        name="test_subagent",
        description="A test subagent",
        system_instructions="You are a subagent",
    )
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        mcp_servers=[mcp_server],
        subagents=[subagent],
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._mcp_servers, [mcp_server])
    self.assertEqual(strategy._subagents, [subagent])

  def test_litert_logging_defaults_to_silent(self):
    """Verify LiteRT logging defaults to SILENT (non-verbose)."""
    with (
        mock.patch.object(litert_connection, "litert_lm") as mock_litert_lm,
        mock.patch.object(logging.Logger, "isEnabledFor", return_value=False),
    ):
      mock_litert_lm.LogSeverity.SILENT = 1000
      mock_litert_lm.set_min_log_severity = mock.MagicMock()
      config = litert_connection_config.LiteRTAgentConfig(
          model_path="/tmp/model.litertlm",
      )
      config.create_strategy(
          tool_runner=mock.MagicMock(),
          hook_runner=mock.MagicMock(),
      )
      mock_litert_lm.set_min_log_severity.assert_called_with(1000)

  def test_litert_logging_defaults_to_silent_with_ambient_debug_logger(self):
    """Verify LiteRT logging is SILENT when debug check is isolated despite ambient debug loggers."""
    root_logger = logging.getLogger()
    old_level = root_logger.level
    try:
      root_logger.setLevel(logging.DEBUG)
      with (
          mock.patch.object(litert_connection, "litert_lm") as mock_litert_lm,
          mock.patch.object(logging.Logger, "isEnabledFor", return_value=False),
      ):
        mock_litert_lm.LogSeverity.SILENT = 1000
        mock_litert_lm.set_min_log_severity = mock.MagicMock()
        config = litert_connection_config.LiteRTAgentConfig(
            model_path="/tmp/model.litertlm",
        )
        config.create_strategy(
            tool_runner=mock.MagicMock(),
            hook_runner=mock.MagicMock(),
        )
        mock_litert_lm.set_min_log_severity.assert_called_with(1000)
    finally:
      root_logger.setLevel(old_level)

  def test_litert_logging_verbose_when_debug_enabled(self):
    """Verify LiteRT log severity is VERBOSE when Python debug logging is enabled."""
    with mock.patch.object(litert_connection, "litert_lm") as mock_litert_lm:
      mock_litert_lm.LogSeverity.VERBOSE = 0
      mock_litert_lm.set_min_log_severity = mock.MagicMock()
      logger = logging.getLogger("google.antigravity")
      old_level = logger.level
      try:
        logger.setLevel(logging.DEBUG)
        config = litert_connection_config.LiteRTAgentConfig(
            model_path="/tmp/model.litertlm",
        )
        config.create_strategy(
            tool_runner=mock.MagicMock(),
            hook_runner=mock.MagicMock(),
        )
        mock_litert_lm.set_min_log_severity.assert_called_with(0)
      finally:
        logger.setLevel(old_level)

  async def test_litert_warmup_timeout_scaling_and_engine_lock_wait(self):
    """Verify warmup timeout scaling and engine lock wait on timeout."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/dummy/path.litertlm",
        compaction_config=types.CompactionConfig(token_threshold=65536),
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    mock_engine_cls = mock.MagicMock()
    mock_engine_inst = mock.MagicMock()
    mock_engine_cls.return_value = mock_engine_inst
    mock_engine_inst.__enter__.return_value = mock.MagicMock()

    lock_acquired = False

    class FakeLock:

      def __enter__(self):
        nonlocal lock_acquired
        lock_acquired = True
        return self

      def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    fake_server_inst = mock.MagicMock()
    fake_server_inst.engine_lock = FakeLock()

    with (
        mock.patch.object(litert_connection, "litert_lm") as mock_lm,
        mock.patch.object(
            litert_connection.litert_server,
            "LiteRTOpenAIServer",
            return_value=fake_server_inst,
        ),
        mock.patch.object(
            litert_connection, "_urlopen_no_proxy"
        ) as mock_urlopen,
        mock.patch("os.path.exists", return_value=True),
        mock.patch(
            "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__",
            return_value=None,
        ),
    ):

      def fake_urlopen(req, timeout=None):
        url_str = str(req.full_url if hasattr(req, "full_url") else str(req))
        if "chat/completions" in url_str:
          # Check scaled timeout: 65536 / 250 = 262.144
          self.assertAlmostEqual(timeout, 262.144, places=2)
          raise TimeoutError("Simulated warm-up timeout")
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

      mock_urlopen.side_effect = fake_urlopen
      mock_lm.Engine = mock_engine_cls
      mock_lm.Backend.CPU.return_value = "CPU"

      try:
        await strategy.__aenter__()
      finally:
        await strategy.__aexit__(None, None, None)

      self.assertTrue(
          lock_acquired,
          "Engine lock should be waited on during warmup cleanup.",
      )

  def test_litert_config_create_strategy_import_error(self):
    """Verify LiteRTAgentConfig raises RuntimeError when litert_connection is not available."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
    )
    local_pkg = sys.modules[
        "google.antigravity.connections.local"
    ]
    orig_attr = getattr(local_pkg, "litert_connection", None)
    try:
      if hasattr(local_pkg, "litert_connection"):
        delattr(local_pkg, "litert_connection")
      with mock.patch.dict(
          sys.modules,
          {
              "google.antigravity.connections.local.litert_connection": (
                  None
              )
          },
      ):
        with self.assertRaisesRegex(
            RuntimeError, "LiteRT backend is not available"
        ):
          config.create_strategy(
              tool_runner=mock.MagicMock(),
              hook_runner=mock.MagicMock(),
          )
    finally:
      if orig_attr is not None:
        setattr(local_pkg, "litert_connection", orig_attr)

  def test_litert_server_handle_error_connection_reset(self):
    """Verify handle_error suppresses reset errors and delegates others."""
    server = litert_connection.litert_server.LiteRTOpenAIServer(
        ("127.0.0.1", 0),
        litert_connection.litert_server.LiteRTOpenAIHandler,
        engine=mock.MagicMock(),
        model_name="test-model",
        max_output_tokens=16384,
        thinking_token_budget=8192,
    )
    try:
      with (
          mock.patch("sys.exc_info") as mock_exc_info,
          mock.patch.object(
              socketserver.BaseServer, "handle_error"
          ) as mock_super_handle_error,
      ):
        # 1. ConnectionResetError is suppressed
        err = ConnectionResetError(
            "[WinError 10054] An existing connection was forcibly closed by the"
            " remote host"
        )
        mock_exc_info.return_value = (ConnectionResetError, err, None)
        server.handle_error(mock.MagicMock(), ("127.0.0.1", 12345))
        mock_super_handle_error.assert_not_called()

        # 2. Windows OSError (winerror=10054) is suppressed
        win_err_10054 = OSError()
        win_err_10054.winerror = 10054
        mock_exc_info.return_value = (OSError, win_err_10054, None)
        server.handle_error(mock.MagicMock(), ("127.0.0.1", 12345))
        mock_super_handle_error.assert_not_called()

        # 3. Windows OSError (winerror=10053) is suppressed
        win_err_10053 = OSError()
        win_err_10053.winerror = 10053
        mock_exc_info.return_value = (OSError, win_err_10053, None)
        server.handle_error(mock.MagicMock(), ("127.0.0.1", 12345))
        mock_super_handle_error.assert_not_called()

        # 4. Unrelated error (e.g. ValueError) delegates to super().handle_error
        val_err = ValueError("Unrelated server failure")
        mock_exc_info.return_value = (ValueError, val_err, None)
        req = mock.MagicMock()
        client_addr = ("127.0.0.1", 12345)
        server.handle_error(req, client_addr)
        mock_super_handle_error.assert_called_once_with(req, client_addr)
    finally:
      server.server_close()

  def test_litert_handler_handle_connection_reset(self):
    """Verify LiteRTOpenAIHandler.handle handles reset errors and re-raises others."""
    handler = mock.MagicMock(
        spec=litert_connection.litert_server.LiteRTOpenAIHandler
    )

    # 1. ConnectionResetError sets close_connection = True
    handler.close_connection = False
    with mock.patch.object(
        http.server.BaseHTTPRequestHandler,
        "handle",
        side_effect=ConnectionResetError,
    ):
      litert_connection.litert_server.LiteRTOpenAIHandler.handle(handler)
      self.assertTrue(handler.close_connection)

    # 2. OSError with winerror 10054 sets close_connection = True
    handler.close_connection = False
    win_err_10054 = OSError()
    win_err_10054.winerror = 10054
    with mock.patch.object(
        http.server.BaseHTTPRequestHandler,
        "handle",
        side_effect=win_err_10054,
    ):
      litert_connection.litert_server.LiteRTOpenAIHandler.handle(handler)
      self.assertTrue(handler.close_connection)

    # 3. OSError with winerror 10053 sets close_connection = True
    handler.close_connection = False
    win_err_10053 = OSError()
    win_err_10053.winerror = 10053
    with mock.patch.object(
        http.server.BaseHTTPRequestHandler,
        "handle",
        side_effect=win_err_10053,
    ):
      litert_connection.litert_server.LiteRTOpenAIHandler.handle(handler)
      self.assertTrue(handler.close_connection)

    # 4. Unrelated OSError (winerror=10050) is re-raised
    handler.close_connection = False
    win_err_other = OSError()
    win_err_other.winerror = 10050
    with mock.patch.object(
        http.server.BaseHTTPRequestHandler,
        "handle",
        side_effect=win_err_other,
    ):
      with self.assertRaises(OSError):
        litert_connection.litert_server.LiteRTOpenAIHandler.handle(handler)
      self.assertFalse(handler.close_connection)

    # 5. Non-OSError exception (e.g. ValueError) is re-raised
    handler.close_connection = False
    with mock.patch.object(
        http.server.BaseHTTPRequestHandler,
        "handle",
        side_effect=ValueError("Unexpected"),
    ):
      with self.assertRaises(ValueError):
        litert_connection.litert_server.LiteRTOpenAIHandler.handle(handler)
      self.assertFalse(handler.close_connection)

  def test_litert_server_custom_output_and_thinking_config(self):
    """Verify LiteRTOpenAIServer stores limits and passes them to create_conversation."""
    mock_engine = mock.MagicMock()
    mock_conv = mock.MagicMock()
    mock_engine.create_conversation.return_value.__enter__.return_value = (
        mock_conv
    )

    server = litert_connection.litert_server.LiteRTOpenAIServer(
        ("127.0.0.1", 0),
        litert_connection.litert_server.LiteRTOpenAIHandler,
        engine=mock_engine,
        model_name="test-model",
        max_output_tokens=4096,
        thinking_token_budget=1024,
    )
    self.assertEqual(server.max_output_tokens, 4096)
    self.assertEqual(server.thinking_token_budget, 1024)

    handler = mock.MagicMock()
    handler.server = server
    handler.path = "/v1/chat/completions"
    handler._stream_response = mock.MagicMock()
    handler._handle_synchronous = mock.MagicMock()

    with mock.patch.object(
        litert_connection.litert_server, "litert_lm"
    ) as mock_lm:
      mock_lm.ThinkingConfig = mock.MagicMock(
          return_value="mock_thinking_config"
      )
      mock_lm.ConstrainedDecodingConfig = mock.MagicMock(
          return_value="mock_cdc"
      )
      mock_lm.Message.user = mock.MagicMock(return_value="mock_msg")
      mock_lm.Contents.of = mock.MagicMock(return_value="mock_contents")

      # 1. Request uses limits configured on server
      payload = json.dumps(
          {"messages": [{"role": "user", "content": "hi"}]}
      ).encode("utf-8")
      handler.headers = {"Content-Length": str(len(payload))}
      handler.rfile.read.return_value = payload
      litert_connection.litert_server.LiteRTOpenAIHandler.do_POST(handler)

      mock_engine.create_conversation.assert_called_once()
      _, kwargs = mock_engine.create_conversation.call_args
      self.assertEqual(kwargs["max_output_tokens"], 4096)
      mock_lm.ThinkingConfig.assert_called_with(thinking_token_budget=1024)
      self.assertEqual(kwargs["thinking_config"], "mock_thinking_config")

      # 2. Server with thinking disabled (None)
      server.thinking_token_budget = None
      mock_engine.create_conversation.reset_mock()
      payload = json.dumps(
          {"messages": [{"role": "user", "content": "hi"}]}
      ).encode("utf-8")
      handler.headers = {"Content-Length": str(len(payload))}
      handler.rfile.read.return_value = payload
      litert_connection.litert_server.LiteRTOpenAIHandler.do_POST(handler)

      _, kwargs = mock_engine.create_conversation.call_args
      self.assertEqual(kwargs["max_output_tokens"], 4096)
      self.assertIsNone(kwargs["thinking_config"])

  def test_policies_forwarded_to_strategy_and_harness_config(self):
    """Verify explicit policies are forwarded to LiteRTConnectionStrategy."""
    denies = [
        policy.deny("create_file"),
        policy.deny("edit_file"),
        policy.deny("run_command"),
    ]
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        policies=denies,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._policies, denies)
    h_cfg = strategy._build_harness_config()
    self.assertTrue(h_cfg.HasField("policy_config"))
    self.assertEqual(len(h_cfg.policy_config.rules), 3)

  def test_default_policies_forwarded_to_strategy(self):
    """Verify default confirm_run_command() policy reaches strategy when policies is omitted."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(len(strategy._policies), 2)
    self.assertEqual(strategy._policies[0].tool, "run_command")
    self.assertEqual(strategy._policies[0].decision, policy.Decision.DENY)
    h_cfg = strategy._build_harness_config()
    self.assertTrue(h_cfg.HasField("policy_config"))
    self.assertEqual(len(h_cfg.policy_config.rules), 2)

  def test_budget_and_session_continuation_mode_forwarded_to_strategy(self):
    """Verify budget_config and session_continuation_mode are forwarded."""
    budget = types.BudgetConfig(max_model_calls=5, max_total_tokens=8000)
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        conversation_id="b" * 32,
        session_continuation_mode=types.SessionContinuationMode.RESUME,
        budget_config=budget,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(
        strategy._session_continuation_mode,
        types.SessionContinuationMode.RESUME,
    )
    self.assertEqual(strategy._budget_config, budget)
    h_cfg = strategy._build_harness_config()
    self.assertEqual(
        h_cfg.session_continuation_mode,
        localharness_pb2.HarnessConfig.RESUME,
    )
    self.assertTrue(h_cfg.HasField("budget_config"))
    self.assertEqual(h_cfg.budget_config.max_model_calls, 5)
    self.assertEqual(h_cfg.budget_config.max_total_tokens, 8000)

  def test_tools_forwarded_to_strategy(self):
    """Verify custom tools are forwarded to strategy and included in harness proto."""
    def custom_lookup(query: str) -> str:
      """Looks up a query."""
      return query

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        tools=[custom_lookup],
    )
    t_runner = tool_runner_mod.ToolRunner(
        tools=config._get_all_custom_tools()
    )
    strategy = config.create_strategy(
        tool_runner=t_runner,
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._tools, [custom_lookup])
    h_cfg = strategy._build_harness_config()
    tool_names = [t.name for t in h_cfg.tools]
    self.assertIn("custom_lookup", tool_names)

  def test_openai_tool_equality(self):
    """Verify OpenAITool value equality comparisons."""
    desc1 = {
        "type": "function",
        "function": {"name": "test_tool", "description": "A tool"},
    }
    desc2 = {
        "type": "function",
        "function": {"name": "test_tool", "description": "A tool"},
    }
    desc3 = {
        "type": "function",
        "function": {"name": "other_tool", "description": "Another tool"},
    }
    tool1 = litert_server.OpenAITool(desc1)
    tool2 = litert_server.OpenAITool(desc2)
    tool3 = litert_server.OpenAITool(desc3)

    self.assertEqual(tool1, tool2)
    self.assertNotEqual(tool1, tool3)
    self.assertNotEqual(tool1, desc1)
    self.assertNotEqual(tool1, MockTool())
    self.assertNotEqual(tool1, None)

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("subprocess.Popen")
  @mock.patch.object(
      local_connection.LocalConnectionStrategy,
      "__aenter__",
      return_value=None,
  )
  async def test_conversation_caching_multi_turn_sync(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify active conversation is reused across matching synchronous multi-turn requests."""
    del mock_super_enter, mock_exists
    mock_popen.return_value = mock.MagicMock()
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    # Initial state after warmup (warmup creates conversation 0)
    engine = strategy._engine
    warmup_conv_count = len(engine.conversations)

    # Turn 1: user query
    t1_payload = {
        "model": "model.litertlm",
        "messages": [{"role": "user", "content": "Turn 1 question"}],
        "stream": False,
    }
    req1 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t1_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req1) as r:
      self.assertEqual(r.status, 200)
      data1 = json.loads(r.read().decode("utf-8"))
      self.assertEqual(
          data1["choices"][0]["message"]["content"], "Mocked sync reply."
      )

    # Turn 1 should have initialized a new conversation (cache miss from warmup)
    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    turn1_conv = engine.conversations[-1]
    self.assertFalse(turn1_conv.exited)
    self.assertEqual(len(turn1_conv.sent_prompts), 1)

    # Turn 2: user includes history from Turn 1
    t2_payload = {
        "model": "model.litertlm",
        "messages": [
            {"role": "user", "content": "Turn 1 question"},
            {"role": "assistant", "content": "Mocked sync reply."},
            {"role": "user", "content": "Turn 2 question"},
        ],
        "stream": False,
    }
    req2 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t2_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req2) as r:
      self.assertEqual(r.status, 200)
      data2 = json.loads(r.read().decode("utf-8"))
      self.assertEqual(
          data2["choices"][0]["message"]["content"], "Mocked sync reply."
      )

    # Turn 2 should REUSE turn1_conv: no new conversation created, not exited
    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    self.assertFalse(turn1_conv.exited)
    self.assertEqual(len(turn1_conv.sent_prompts), 2)

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("subprocess.Popen")
  @mock.patch.object(
      local_connection.LocalConnectionStrategy,
      "__aenter__",
      return_value=None,
  )
  async def test_conversation_caching_multi_turn_stream(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify active conversation is reused across streaming multi-turn requests."""
    del mock_super_enter, mock_exists
    mock_popen.return_value = mock.MagicMock()
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    engine = strategy._engine
    warmup_conv_count = len(engine.conversations)

    def _read_sse_stream(r):
      while True:
        line = r.readline().decode("utf-8")
        if not line or "[DONE]" in line:
          break

    # Turn 1: stream
    t1_payload = {
        "model": "model.litertlm",
        "messages": [{"role": "user", "content": "Stream question 1"}],
        "stream": True,
    }
    req1 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t1_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req1) as r:
      self.assertEqual(r.status, 200)
      _read_sse_stream(r)

    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    turn1_conv = engine.conversations[-1]
    self.assertFalse(turn1_conv.exited)

    # Turn 2: stream with history
    t2_payload = {
        "model": "model.litertlm",
        "messages": [
            {"role": "user", "content": "Stream question 1"},
            {"role": "assistant", "content": "Mocked async reply."},
            {"role": "user", "content": "Stream question 2"},
        ],
        "stream": True,
    }
    req2 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t2_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req2) as r:
      self.assertEqual(r.status, 200)
      _read_sse_stream(r)

    # Reused conversation
    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    self.assertFalse(turn1_conv.exited)
    self.assertEqual(len(turn1_conv.sent_prompts), 2)

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("subprocess.Popen")
  @mock.patch.object(
      local_connection.LocalConnectionStrategy,
      "__aenter__",
      return_value=None,
  )
  async def test_conversation_cache_miss_on_diverging_messages(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify cache miss closes previous conversation and creates a new one."""
    del mock_super_enter, mock_exists
    mock_popen.return_value = mock.MagicMock()
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    engine = strategy._engine
    warmup_conv_count = len(engine.conversations)

    # Turn 1
    t1_payload = {
        "model": "model.litertlm",
        "messages": [{"role": "user", "content": "Initial conversation"}],
        "stream": False,
    }
    req1 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t1_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req1) as r:
      r.read()

    turn1_conv = engine.conversations[-1]
    self.assertFalse(turn1_conv.exited)

    # Turn 2 with completely different messages (divergent history)
    t2_payload = {
        "model": "model.litertlm",
        "messages": [{"role": "user", "content": "Completely new topic"}],
        "stream": False,
    }
    req2 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t2_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req2) as r:
      r.read()

    # Previous conversation must be closed, and a new one opened
    self.assertTrue(turn1_conv.exited)
    self.assertEqual(len(engine.conversations), warmup_conv_count + 2)
    turn2_conv = engine.conversations[-1]
    self.assertFalse(turn2_conv.exited)

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("subprocess.Popen")
  @mock.patch.object(
      local_connection.LocalConnectionStrategy,
      "__aenter__",
      return_value=None,
  )
  async def test_conversation_cache_miss_on_tools_change(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify tool configuration change triggers conversation cache invalidation."""
    del mock_super_enter, mock_exists
    mock_popen.return_value = mock.MagicMock()
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    tool_a = {
        "type": "function",
        "function": {"name": "tool_a", "description": "Tool A"},
    }
    tool_b = {
        "type": "function",
        "function": {"name": "tool_b", "description": "Tool B"},
    }

    # Turn 1 with tool_a
    t1_payload = {
        "model": "model.litertlm",
        "messages": [{"role": "user", "content": "Run tool"}],
        "tools": [tool_a],
        "stream": False,
    }
    req1 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t1_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req1) as r:
      r.read()

    turn1_conv = strategy._engine.conversations[-1]
    self.assertFalse(turn1_conv.exited)

    # Turn 2 with same tools: [tool_a] and matching history
    t2_payload = {
        "model": "model.litertlm",
        "messages": [
            {"role": "user", "content": "Run tool"},
            {"role": "assistant", "content": "Mocked sync reply."},
            {"role": "user", "content": "Run tool again"},
        ],
        "tools": [tool_a],
        "stream": False,
    }
    req2 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t2_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req2) as r:
      r.read()

    self.assertFalse(turn1_conv.exited)
    self.assertEqual(len(turn1_conv.sent_prompts), 2)

    # Turn 3 with tool_b added -> cache miss invalidates turn1_conv
    t3_payload = {
        "model": "model.litertlm",
        "messages": [
            {"role": "user", "content": "Run tool"},
            {"role": "assistant", "content": "Mocked sync reply."},
            {"role": "user", "content": "Run tool again"},
            {"role": "assistant", "content": "Mocked sync reply."},
            {"role": "user", "content": "Run another tool"},
        ],
        "tools": [tool_a, tool_b],
        "stream": False,
    }
    req3 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t3_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req3) as r:
      r.read()

    # Tools changed -> turn1_conv closed, new conv created
    self.assertTrue(turn1_conv.exited)
    turn3_conv = strategy._engine.conversations[-1]
    self.assertNotEqual(turn3_conv, turn1_conv)
    self.assertFalse(turn3_conv.exited)

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("subprocess.Popen")
  @mock.patch.object(
      local_connection.LocalConnectionStrategy,
      "__aenter__",
      return_value=None,
  )
  async def test_consecutive_tool_responses_merged_and_cached(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify consecutive tool responses are merged into a single tool message and reuse cache."""
    del mock_super_enter, mock_exists
    mock_popen.return_value = mock.MagicMock()
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    engine = strategy._engine
    warmup_conv_count = len(engine.conversations)

    # Setup Turn 1 simulated response with 2 tool calls
    tool_calls_data = [
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        {"function": {"name": "get_time", "arguments": {"city": "Paris"}}},
    ]
    t1_payload = {
        "model": "model.litertlm",
        "messages": [{"role": "user", "content": "Weather and time"}],
        "stream": False,
    }
    req1 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t1_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    def _inject_tool_calls(messages=None, tools=None, **kwargs):
      conv = MockConversation(messages, tools, **kwargs)
      conv.simulated_response = {
          "content": [],
          "tool_calls": tool_calls_data,
      }
      engine.conversations.append(conv)
      return conv

    with mock.patch.object(
        engine, "create_conversation", side_effect=_inject_tool_calls
    ):
      with _urlopen_no_proxy(req1) as r:
        self.assertEqual(r.status, 200)
        data1 = json.loads(r.read().decode("utf-8"))
        tc_resp = data1["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(tc_resp), 2)
        call_id_1 = tc_resp[0]["id"]
        call_id_2 = tc_resp[1]["id"]

    turn1_conv = engine.conversations[-1]
    self.assertFalse(turn1_conv.exited)
    # Set simulated response for Turn 2
    turn1_conv.simulated_response = {
        "content": [{"type": "text", "text": "Mocked sync reply."}],
        "tool_calls": [],
    }

    # Turn 2: Echo assistant message with tool calls, followed by 2
    # consecutive tool responses.
    t2_payload = {
        "model": "model.litertlm",
        "messages": [
            {"role": "user", "content": "Weather and time"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id_1,
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    },
                    {
                        "id": call_id_2,
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"city": "Paris"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id_1,
                "content": "Sunny, 20C",
            },
            {
                "role": "tool",
                "tool_call_id": call_id_2,
                "content": "14:00",
            },
        ],
        "stream": False,
    }
    req2 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t2_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req2) as r:
      self.assertEqual(r.status, 200)
      data2 = json.loads(r.read().decode("utf-8"))
      self.assertEqual(
          data2["choices"][0]["message"]["content"], "Mocked sync reply."
      )

    # Turn 2 must reuse turn1_conv
    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    self.assertFalse(turn1_conv.exited)
    self.assertEqual(len(turn1_conv.sent_prompts), 2)

    # Verify merged tool response prompt
    merged_prompt = turn1_conv.sent_prompts[-1]
    self.assertEqual(merged_prompt["role"], "tool")
    contents = merged_prompt["content"].contents
    self.assertEqual(len(contents), 2)
    self.assertEqual(contents[0].name, "get_weather")
    self.assertEqual(contents[0].response, "Sunny, 20C")
    self.assertEqual(contents[1].name, "get_time")
    self.assertEqual(contents[1].response, "14:00")

    # Turn 3: Follow-up user turn reusing the conversation after tool execution.
    t3_payload = {
        "model": "model.litertlm",
        "messages": [
            *t2_payload["messages"],
            {"role": "assistant", "content": "Mocked sync reply."},
            {"role": "user", "content": "Thanks!"},
        ],
        "stream": False,
    }
    req3 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t3_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req3) as r:
      self.assertEqual(r.status, 200)
      data3 = json.loads(r.read().decode("utf-8"))
      self.assertEqual(
          data3["choices"][0]["message"]["content"], "Mocked sync reply."
      )

    # Turn 3 must also reuse turn1_conv
    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    self.assertFalse(turn1_conv.exited)
    self.assertEqual(len(turn1_conv.sent_prompts), 3)

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("subprocess.Popen")
  @mock.patch.object(
      local_connection.LocalConnectionStrategy,
      "__aenter__",
      return_value=None,
  )
  async def test_conversation_caching_multi_turn_stream_with_tool_calls(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify multi-turn streaming tool call response reuses cached conversation."""
    del mock_super_enter, mock_exists
    mock_popen.return_value = mock.MagicMock()
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    engine = strategy._engine
    warmup_conv_count = len(engine.conversations)

    tool_calls_data = [
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
    ]

    def _inject_tool_calls(messages=None, tools=None, **kwargs):
      conv = MockConversation(messages, tools, **kwargs)
      conv.simulated_async_response = [
          {"content": [], "tool_calls": tool_calls_data}
      ]
      engine.conversations.append(conv)
      return conv

    # Turn 1: Stream tool call
    t1_payload = {
        "model": "model.litertlm",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "stream": True,
    }
    req1 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t1_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    tool_call_id = None
    with mock.patch.object(
        engine, "create_conversation", side_effect=_inject_tool_calls
    ):
      with _urlopen_no_proxy(req1) as r:
        self.assertEqual(r.status, 200)
        while True:
          line = r.readline().decode("utf-8")
          if not line or "[DONE]" in line:
            break
          if line.startswith("data: "):
            chunk_data = json.loads(line[6:].strip())
            choices = chunk_data.get("choices", [])
            if choices:
              delta = choices[0].get("delta", {})
              if "tool_calls" in delta and delta["tool_calls"]:
                tool_call_id = delta["tool_calls"][0]["id"]

    self.assertIsNotNone(tool_call_id)
    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    turn1_conv = engine.conversations[-1]
    self.assertFalse(turn1_conv.exited)

    # Reset simulated async response to default text streaming
    turn1_conv.simulated_async_response = None

    # Turn 2: Echo assistant tool call + tool response, stream text reply
    t2_payload = {
        "model": "model.litertlm",
        "messages": [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "Sunny, 20C",
            },
        ],
        "stream": True,
    }
    req2 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t2_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req2) as r:
      self.assertEqual(r.status, 200)
      while True:
        line = r.readline().decode("utf-8")
        if not line or "[DONE]" in line:
          break

    # Turn 2 must reuse turn1_conv
    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    self.assertFalse(turn1_conv.exited)
    self.assertEqual(len(turn1_conv.sent_prompts), 2)

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("subprocess.Popen")
  @mock.patch.object(
      local_connection.LocalConnectionStrategy,
      "__aenter__",
      return_value=None,
  )
  async def test_client_disconnect_cancels_and_closes_cached_conversation(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify client disconnect or stream error cancels and resets cached conversation."""
    del mock_super_enter, mock_exists
    mock_popen.return_value = mock.MagicMock()
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    server = strategy._openai_server

    def _failing_stream(_):
      yield {"content": [{"type": "text", "text": "Part 1"}]}
      raise BrokenPipeError("Simulated client disconnect")

    engine = strategy._engine

    def _create_err_conv(messages=None, tools=None, **kwargs):
      conv = MockConversation(messages, tools, **kwargs)
      conv.send_message_async = _failing_stream
      engine.conversations.append(conv)
      return conv

    with mock.patch.object(
        engine, "create_conversation", side_effect=_create_err_conv
    ):
      t_payload = {
          "model": "model.litertlm",
          "messages": [{"role": "user", "content": "Abort stream"}],
          "stream": True,
      }
      req = urllib.request.Request(
          f"{strategy._openai_server_url}/v1/chat/completions",
          data=json.dumps(t_payload).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with _urlopen_no_proxy(req) as r:
        self.assertEqual(r.status, 200)
        # Consume stream until connection aborts on server side
        while r.readline():
          pass

    failed_conv = engine.conversations[-1]
    self.assertTrue(failed_conv.cancelled)
    self.assertTrue(failed_conv.exited)
    self.assertIsNone(server.litert_lm_conversation)
    self.assertIsNone(server.conversation_cache_key)

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("subprocess.Popen")
  @mock.patch.object(
      local_connection.LocalConnectionStrategy,
      "__aenter__",
      return_value=None,
  )
  async def test_strategy_shutdown_closes_cached_conversation(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify shutting down strategy closes cached conversation under lock."""
    del mock_super_enter, mock_exists
    mock_popen.return_value = mock.MagicMock()
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    await strategy.__aenter__()

    # Execute a turn to ensure conversation is cached
    t_payload = {
        "model": "model.litertlm",
        "messages": [{"role": "user", "content": "Hello before shutdown"}],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req) as r:
      r.read()

    server = strategy._openai_server
    self.assertIsNotNone(server.litert_lm_conversation)
    cached_conv = server.litert_lm_conversation
    self.assertFalse(cached_conv.exited)

    # Exit strategy
    await strategy.__aexit__(None, None, None)

    # Conversation must be closed, server cleared
    self.assertTrue(cached_conv.exited)
    self.assertIsNone(strategy._openai_server)

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("subprocess.Popen")
  @mock.patch.object(
      local_connection.LocalConnectionStrategy,
      "__aenter__",
      return_value=None,
  )
  async def test_thought_channel_streaming_and_sync_with_cache_reuse(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify thought channel is emitted as reasoning and reasoning_content."""
    del mock_super_enter, mock_exists
    mock_popen.return_value = mock.MagicMock()
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    engine = strategy._engine
    warmup_conv_count = len(engine.conversations)

    def _inject_thought_chunks(messages=None, tools=None, **kwargs):
      conv = MockConversation(messages, tools, **kwargs)
      conv.simulated_async_response = [
          {"channels": {"thought": "Step 1: analyze riddle. "}},
          {"channels": {"thought": "Step 2: wind -> echo."}},
          {"content": [{"type": "text", "text": "An echo."}]},
      ]
      conv.simulated_response = {
          "channels": {"thought": "Sync reasoning."},
          "content": [{"type": "text", "text": "Sync answer."}],
      }
      engine.conversations.append(conv)
      return conv

    # 1. Turn 1 (Streaming): Verify reasoning and content deltas.
    t1_payload = {
        "model": "model.litertlm",
        "messages": [{"role": "user", "content": "Solve this riddle"}],
        "stream": True,
    }
    req1 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t1_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    reasoning_deltas = []
    reasoning_content_deltas = []
    content_deltas = []
    with mock.patch.object(
        engine, "create_conversation", side_effect=_inject_thought_chunks
    ):
      with _urlopen_no_proxy(req1) as r:
        self.assertEqual(r.status, 200)
        while True:
          line = r.readline().decode("utf-8")
          if not line or "[DONE]" in line:
            break
          if line.startswith("data: "):
            chunk_data = json.loads(line[6:].strip())
            choices = chunk_data.get("choices", [])
            if choices:
              delta = choices[0].get("delta", {})
              if "reasoning" in delta:
                reasoning_deltas.append(delta["reasoning"])
              if "reasoning_content" in delta:
                reasoning_content_deltas.append(delta["reasoning_content"])
              if "content" in delta:
                content_deltas.append(delta["content"])

    self.assertEqual(
        "".join(reasoning_deltas),
        "Step 1: analyze riddle. Step 2: wind -> echo.",
    )
    self.assertEqual(
        "".join(reasoning_content_deltas),
        "Step 1: analyze riddle. Step 2: wind -> echo.",
    )
    self.assertEqual("".join(content_deltas), "An echo.")
    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    turn1_conv = engine.conversations[-1]

    # 2. Turn 2 (Synchronous): Verify cache hit and sync reasoning fields.
    t2_payload = {
        "model": "model.litertlm",
        "messages": [
            {"role": "user", "content": "Solve this riddle"},
            {"role": "assistant", "content": "An echo."},
            {"role": "user", "content": "Are you sure?"},
        ],
        "stream": False,
    }
    req2 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(t2_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req2) as r:
      self.assertEqual(r.status, 200)
      data2 = json.loads(r.read().decode("utf-8"))
      msg2 = data2["choices"][0]["message"]
      self.assertEqual(msg2["reasoning"], "Sync reasoning.")
      self.assertEqual(msg2["reasoning_content"], "Sync reasoning.")
      self.assertEqual(msg2["content"], "Sync answer.")

    # Must have reused turn1_conv
    self.assertEqual(len(engine.conversations), warmup_conv_count + 1)
    self.assertFalse(turn1_conv.exited)
    self.assertEqual(len(turn1_conv.sent_prompts), 2)

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_multimodal_image_url_base64(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify image_url with data: base64 translates to ImageBytes."""
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(), hook_runner=mock.MagicMock()
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    raw_bytes = b"jpeg_image_data_test"
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    payload = {
        "model": "model.litertlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this photo:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"},
                },
            ],
        }],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req) as r:
      self.assertEqual(r.status, 200)

    conv = strategy._engine.conversations[-1]
    prompt = conv.sent_prompts[-1]
    self.assertEqual(prompt["role"], "user")
    contents = prompt["content"].contents
    self.assertEqual(len(contents), 2)
    self.assertIsInstance(contents[0], MockText)
    self.assertEqual(contents[0].text, "Describe this photo:")
    self.assertIsInstance(contents[1], MockImageBytes)
    self.assertEqual(contents[1].bytes, raw_bytes)

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_multimodal_image_url_file_and_raw_path(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify image_url with file:// and raw path translates to ImageFile."""
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(), hook_runner=mock.MagicMock()
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    payload = {
        "model": "model.litertlm",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "file:///path/to/diagram.png"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "/path/to/second_diagram.png"},
                },
            ],
        }],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req) as r:
      self.assertEqual(r.status, 200)

    conv = strategy._engine.conversations[-1]
    prompt = conv.sent_prompts[-1]
    contents = prompt["content"].contents
    self.assertEqual(len(contents), 2)
    self.assertIsInstance(contents[0], MockImageFile)
    self.assertEqual(contents[0].absolute_path, "/path/to/diagram.png")
    self.assertIsInstance(contents[1], MockImageFile)
    self.assertEqual(contents[1].absolute_path, "/path/to/second_diagram.png")

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_multimodal_image_blob_and_path(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify image part with blob and path translates to ImageBytes and ImageFile."""
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(), hook_runner=mock.MagicMock()
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    raw_bytes = b"blob_image_data"
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    payload = {
        "model": "model.litertlm",
        "messages": [{
            "role": "user",
            "content": [
                "Compare these two images:",
                {"type": "image", "blob": b64_str},
                {"type": "image", "path": "/media/chart.png"},
            ],
        }],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req) as r:
      self.assertEqual(r.status, 200)

    conv = strategy._engine.conversations[-1]
    prompt = conv.sent_prompts[-1]
    contents = prompt["content"].contents
    self.assertEqual(len(contents), 3)
    self.assertIsInstance(contents[0], MockText)
    self.assertEqual(contents[0].text, "Compare these two images:")
    self.assertIsInstance(contents[1], MockImageBytes)
    self.assertEqual(contents[1].bytes, raw_bytes)
    self.assertIsInstance(contents[2], MockImageFile)
    self.assertEqual(contents[2].absolute_path, "/media/chart.png")

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_multimodal_audio_input(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify input_audio and audio parts translate to AudioBytes and AudioFile."""
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(), hook_runner=mock.MagicMock()
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    raw_audio = b"pcm_audio_bytes"
    b64_str = base64.b64encode(raw_audio).decode("utf-8")
    payload = {
        "model": "model.litertlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": b64_str}},
                {"type": "audio", "blob": b64_str},
                {"type": "audio", "path": "/audio/voice.wav"},
            ],
        }],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req) as r:
      self.assertEqual(r.status, 200)

    conv = strategy._engine.conversations[-1]
    prompt = conv.sent_prompts[-1]
    contents = prompt["content"].contents
    self.assertEqual(len(contents), 3)
    self.assertIsInstance(contents[0], MockAudioBytes)
    self.assertEqual(contents[0].bytes, raw_audio)
    self.assertIsInstance(contents[1], MockAudioBytes)
    self.assertEqual(contents[1].bytes, raw_audio)
    self.assertIsInstance(contents[2], MockAudioFile)
    self.assertEqual(contents[2].absolute_path, "/audio/voice.wav")

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_multimodal_error_handling(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify malformed and invalid multimodal parts return 400 Bad Request."""
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(), hook_runner=mock.MagicMock()
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    invalid_contents = [
        # Data URL missing comma
        [{"type": "image_url", "image_url": {"url": "data:image/jpeg"}}],
        # Data URL missing base64
        [{
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;charset=utf-8,hello"},
        }],
        # Empty image_url
        [{"type": "image_url", "image_url": {"url": ""}}],
        # Corrupted base64
        [{
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,not-valid-b64-!"},
        }],
        # Image missing blob and path
        [{"type": "image"}],
        # input_audio missing data
        [{"type": "input_audio", "input_audio": {}}],
        # audio missing blob and path
        [{"type": "audio"}],
        # Non-str / non-dict content part element
        [123],
        # Unsupported part type
        [{"type": "video_stream", "url": "http://stream.mp4"}],
    ]

    for content_part in invalid_contents:
      payload = {
          "model": "model.litertlm",
          "messages": [{"role": "user", "content": content_part}],
          "stream": False,
      }
      req = urllib.request.Request(
          f"{strategy._openai_server_url}/v1/chat/completions",
          data=json.dumps(payload).encode("utf-8"),
          headers={"Content-Type": "application/json"},
      )
      with self.assertRaises(urllib.error.HTTPError) as ctx:
        with _urlopen_no_proxy(req):
          pass
      self.assertEqual(
          ctx.exception.code, 400, f"Expected 400 for {content_part}"
      )

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_multimodal_conversation_caching_and_reuse(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify multi-turn conversation caching hits and reuses conversation with multimodal messages."""
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(), hook_runner=mock.MagicMock()
    )
    await strategy.__aenter__()
    self.addAsyncCleanup(strategy.__aexit__, None, None, None)

    warmup_conv_count = len(strategy._engine.conversations)

    raw_bytes = b"multimodal_turn_image"
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    system_list_content = [
        {"type": "text", "text": "You are a helpful vision assistant."}
    ]
    user_multimodal_content = [
        {"type": "text", "text": "What is in this diagram?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_str}"},
        },
    ]

    # Turn 1: user sends multimodal prompt
    payload_t1 = {
        "model": "model.litertlm",
        "messages": [
            {"role": "system", "content": system_list_content},
            {"role": "user", "content": user_multimodal_content},
        ],
        "stream": False,
    }
    req1 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload_t1).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req1) as r:
      self.assertEqual(r.status, 200)

    self.assertEqual(len(strategy._engine.conversations), warmup_conv_count + 1)
    conv1 = strategy._engine.conversations[-1]
    self.assertEqual(len(conv1.sent_prompts), 1)

    # Turn 2: send turn 1 history + assistant response + new user turn
    payload_t2 = {
        "model": "model.litertlm",
        "messages": [
            {"role": "system", "content": system_list_content},
            {"role": "user", "content": user_multimodal_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Mocked sync reply."}],
            },
            {"role": "user", "content": "Can you summarize it in 3 words?"},
        ],
        "stream": False,
    }
    req2 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload_t2).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req2) as r:
      self.assertEqual(r.status, 200)

    # Verify conversation cache hit: no new conversation created
    self.assertEqual(
        len(strategy._engine.conversations),
        warmup_conv_count + 1,
        "Expected conversation cache HIT to reuse existing conversation",
    )
    self.assertEqual(len(conv1.sent_prompts), 2)
    self.assertEqual(
        conv1.sent_prompts[1]["content"], "Can you summarize it in 3 words?"
    )

    # Turn 3: send different context (modified image) -> cache MISS
    different_b64 = base64.b64encode(b"different_image_bytes").decode("utf-8")
    payload_t3 = {
        "model": "model.litertlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this diagram?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{different_b64}"
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "Mocked sync reply."},
            {"role": "user", "content": "Different follow-up question"},
        ],
        "stream": False,
    }
    req3 = urllib.request.Request(
        f"{strategy._openai_server_url}/v1/chat/completions",
        data=json.dumps(payload_t3).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _urlopen_no_proxy(req3) as r:
      self.assertEqual(r.status, 200)

    # Verify conversation cache miss: new conversation created
    self.assertEqual(
        len(strategy._engine.conversations),
        warmup_conv_count + 2,
        "Expected conversation cache MISS to create a new conversation",
    )

  def test_detect_default_vision_backend(self):
    """Verify _detect_default_vision_backend correctly identifies vision modalities."""
    # 1. Model without vision returns None
    with mock.patch.object(
        litert_connection.litert_lm, "ModelInfo"
    ) as mock_info:
      mock_instance = mock.MagicMock()
      mock_instance.__enter__.return_value = mock_instance
      mock_instance.input_modalities.vision = False
      mock_info.return_value = mock_instance
      self.assertIsNone(
          litert_connection._detect_default_vision_backend(
              "/models/text_only.litertlm"
          )
      )

    # 2. Model with vision and GPU supported backend
    with mock.patch.object(
        litert_connection.litert_lm, "ModelInfo"
    ) as mock_info:
      mock_instance = mock.MagicMock()
      mock_instance.__enter__.return_value = mock_instance
      mock_instance.input_modalities.vision = True
      mock_instance.supported_backends_for_modality.return_value = [
          "gpu",
          "cpu",
      ]
      mock_info.return_value = mock_instance
      self.assertEqual(
          litert_connection._detect_default_vision_backend(
              "/models/vision_gpu.litertlm"
          ),
          litert_connection_config.LiteRTBackend.GPU,
      )

    # 3. Model with vision and non-CPU/GPU (e.g. npu) supported backend -> falls back to CPU
    with mock.patch.object(
        litert_connection.litert_lm, "ModelInfo"
    ) as mock_info:
      mock_instance = mock.MagicMock()
      mock_instance.__enter__.return_value = mock_instance
      mock_instance.input_modalities.vision = True
      mock_instance.supported_backends_for_modality.return_value = ["npu"]
      mock_info.return_value = mock_instance
      self.assertEqual(
          litert_connection._detect_default_vision_backend(
              "/models/vision_npu.litertlm"
          ),
          litert_connection_config.LiteRTBackend.CPU,
      )

    # 4. Model with vision and ["npu", "gpu"] -> selects GPU
    with mock.patch.object(
        litert_connection.litert_lm, "ModelInfo"
    ) as mock_info:
      mock_instance = mock.MagicMock()
      mock_instance.__enter__.return_value = mock_instance
      mock_instance.input_modalities.vision = True
      mock_instance.supported_backends_for_modality.return_value = [
          "npu",
          "gpu",
      ]
      mock_info.return_value = mock_instance
      self.assertEqual(
          litert_connection._detect_default_vision_backend(
              "/models/vision_npu_gpu.litertlm"
          ),
          litert_connection_config.LiteRTBackend.GPU,
      )

    # 5. Model with vision and CPU supported backend
    with mock.patch.object(
        litert_connection.litert_lm, "ModelInfo"
    ) as mock_info:
      mock_instance = mock.MagicMock()
      mock_instance.__enter__.return_value = mock_instance
      mock_instance.input_modalities.vision = True
      mock_instance.supported_backends_for_modality.return_value = ["cpu"]
      mock_info.return_value = mock_instance
      self.assertEqual(
          litert_connection._detect_default_vision_backend(
              "/models/vision_cpu.litertlm"
          ),
          litert_connection_config.LiteRTBackend.CPU,
      )

    # 6. Model with vision but supported_backends raises or is empty -> defaults to CPU
    with mock.patch.object(
        litert_connection.litert_lm, "ModelInfo"
    ) as mock_info:
      mock_instance = mock.MagicMock()
      mock_instance.__enter__.return_value = mock_instance
      mock_instance.input_modalities.vision = True
      mock_instance.supported_backends_for_modality.side_effect = RuntimeError(
          "Failed"
      )
      mock_info.return_value = mock_instance
      self.assertEqual(
          litert_connection._detect_default_vision_backend(
              "/models/vision_fallback.litertlm"
          ),
          litert_connection_config.LiteRTBackend.CPU,
      )

    # 7. ModelInfo initialization failure raises exception -> returns None
    with mock.patch.object(
        litert_connection.litert_lm,
        "ModelInfo",
        side_effect=RuntimeError("Corrupt file or section error"),
    ):
      self.assertIsNone(
          litert_connection._detect_default_vision_backend(
              "/models/corrupt.litertlm"
          )
      )

    # 8. litert_lm is None -> returns None
    with mock.patch.object(litert_connection, "litert_lm", None):
      self.assertIsNone(
          litert_connection._detect_default_vision_backend(
              "/models/any.litertlm"
          )
      )

    # 9. litert_lm missing ModelInfo attribute -> returns None
    mock_litert_without_model_info = mock.MagicMock(spec=[])
    with mock.patch.object(
        litert_connection, "litert_lm", mock_litert_without_model_info
    ):
      self.assertIsNone(
          litert_connection._detect_default_vision_backend(
              "/models/any.litertlm"
          )
      )

    # 10. ModelInfo context manager cleans up (__exit__ is invoked)
    context_mock = MockModelInfo("/models/ctx.litertlm")
    context_mock.input_modalities.vision = True
    with mock.patch.object(
        litert_connection.litert_lm, "ModelInfo", return_value=context_mock
    ):
      backend = litert_connection._detect_default_vision_backend(
          "/models/ctx.litertlm"
      )
      self.assertEqual(backend, litert_connection_config.LiteRTBackend.CPU)
      self.assertTrue(context_mock.closed)

  @mock.patch("os.path.exists")
  @mock.patch("subprocess.Popen")
  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__"
  )
  async def test_vision_backend_auto_detection_and_override_in_aenter(
      self, mock_super_enter, mock_popen, mock_exists
  ):
    """Verify vision_backend auto-detection in __aenter__ and precedence."""
    mock_exists.return_value = True
    mock_super_enter.return_value = None
    mock_popen.return_value = mock.MagicMock()

    # Case 1: Model supports vision, vision_backend is None -> auto-detected
    # as GPU
    with mock.patch.object(
        litert_connection, "_detect_default_vision_backend"
    ) as mock_detect:
      mock_detect.return_value = litert_connection_config.LiteRTBackend.GPU
      config = litert_connection_config.LiteRTAgentConfig(
          model_path="/tmp/vision_model.litertlm",
          backend=litert_connection_config.LiteRTBackend.CPU,
          vision_backend=None,
      )
      strategy = config.create_strategy(
          tool_runner=mock.MagicMock(), hook_runner=mock.MagicMock()
      )
      await strategy.__aenter__()
      self.addAsyncCleanup(strategy.__aexit__, None, None, None)
      self.assertEqual(strategy._engine.kwargs["vision_backend"], "gpu")

    # Case 2: Explicit vision_backend override takes precedence over
    # auto-detection
    with mock.patch.object(
        litert_connection, "_detect_default_vision_backend"
    ) as mock_detect:
      mock_detect.return_value = litert_connection_config.LiteRTBackend.CPU
      config = litert_connection_config.LiteRTAgentConfig(
          model_path="/tmp/vision_model.litertlm",
          backend=litert_connection_config.LiteRTBackend.CPU,
          vision_backend=litert_connection_config.LiteRTBackend.GPU,
      )
      strategy = config.create_strategy(
          tool_runner=mock.MagicMock(), hook_runner=mock.MagicMock()
      )
      await strategy.__aenter__()
      self.addAsyncCleanup(strategy.__aexit__, None, None, None)
      self.assertEqual(strategy._engine.kwargs["vision_backend"], "gpu")

    # Case 3: Model does not support vision and vision_backend is None ->
    # engine vision_backend is None
    with mock.patch.object(
        litert_connection, "_detect_default_vision_backend"
    ) as mock_detect:
      mock_detect.return_value = None
      config = litert_connection_config.LiteRTAgentConfig(
          model_path="/tmp/text_only_model.litertlm",
          backend=litert_connection_config.LiteRTBackend.CPU,
          vision_backend=None,
      )
      strategy = config.create_strategy(
          tool_runner=mock.MagicMock(), hook_runner=mock.MagicMock()
      )
      await strategy.__aenter__()
      self.addAsyncCleanup(strategy.__aexit__, None, None, None)
      self.assertIsNone(strategy._engine.kwargs["vision_backend"])


class DeriveLiteRTCompactionConfigTest(unittest.TestCase):

  def test_derive_default_tokens(self):
    """Verify ceiling and interval derivation with standard defaults."""
    config = litert_connection_config.derive_litert_compaction_config()
    self.assertEqual(config.token_threshold, 40960)

    config_explicit = litert_connection_config.derive_litert_compaction_config(
        max_kv_cache_tokens=65536,
    )
    self.assertEqual(config_explicit.token_threshold, 40960)

  def test_derive_custom_tokens(self):
    """Verify ceiling and interval derivation with custom parameters."""
    config = litert_connection_config.derive_litert_compaction_config(
        max_kv_cache_tokens=32768,
        max_output_tokens=8192,
    )
    self.assertEqual(config.token_threshold, 16384)

  def test_derive_floor_clamp(self):
    """Verify context ceiling is clamped to at least 1024 tokens."""
    config = litert_connection_config.derive_litert_compaction_config(
        max_kv_cache_tokens=10000,
        max_output_tokens=5000,
    )
    self.assertEqual(config.token_threshold, 1024)

    config = litert_connection_config.derive_litert_compaction_config(
        max_kv_cache_tokens=2048,
        max_output_tokens=2048,
    )
    self.assertEqual(config.token_threshold, 1024)

  def test_derive_non_positive_kv_cache_tokens_raises(self):
    """Verify ValueError is raised if max_kv_cache_tokens <= 0."""
    with self.assertRaises(ValueError):
      litert_connection_config.derive_litert_compaction_config(
          max_kv_cache_tokens=0,
          max_output_tokens=16384,
      )
    with self.assertRaises(ValueError):
      litert_connection_config.derive_litert_compaction_config(
          max_kv_cache_tokens=-1000,
          max_output_tokens=16384,
      )

  def test_derive_non_positive_output_tokens_raises(self):
    """Verify ValueError is raised if max_output_tokens <= 0."""
    with self.assertRaises(ValueError):
      litert_connection_config.derive_litert_compaction_config(
          max_kv_cache_tokens=65536,
          max_output_tokens=0,
      )
    with self.assertRaises(ValueError):
      litert_connection_config.derive_litert_compaction_config(
          max_kv_cache_tokens=65536,
          max_output_tokens=-2000,
      )


if __name__ == "__main__":
  unittest.main()
