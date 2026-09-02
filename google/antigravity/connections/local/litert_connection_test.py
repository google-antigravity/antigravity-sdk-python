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

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

  def create_conversation(self, messages=None, tools=None, **kwargs):
    return MockConversation(messages, tools, **kwargs)


class MockConversation:

  def __init__(self, messages, tools=None, **kwargs):
    self.messages = messages or []
    self.tools = tools or []
    self.kwargs = kwargs
    self.simulated_response = None
    self.simulated_async_response = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

  # pylint: disable=unused-argument
  def send_message(self, prompt):
    if self.simulated_response is not None:
      return self.simulated_response
    return {"content": [{"type": "text", "text": "Mocked sync reply."}]}

  # pylint: disable=unused-argument
  def send_message_async(self, prompt):
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


class MockContents:

  def __init__(self, contents):
    self.contents = contents

  @classmethod
  def of(cls, *args):
    if not args:
      return cls([])
    arg = args[0]
    if isinstance(arg, str):
      return cls([MockText(arg)])
    return cls([arg])

  @classmethod
  def empty(cls):
    return cls([])


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


class MockTool:

  def get_tool_description(self) -> dict[str, Any]:
    return {}

  def execute(self, param: Any) -> Any:
    del param  # Unused
    return None


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
mock_litert.Content.ToolResponse = MockToolResponse
mock_litert.Tool = MockTool
mock_litert.ToolCall = MockToolCall

sys.modules["litert_lm"] = mock_litert

# pylint: disable=g-import-not-at-top
from google.antigravity.proto import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local import litert_connection
from google.antigravity.connections.local import litert_connection_config
from google.antigravity.connections.local import litert_server
from google.antigravity.connections.local import local_connection
from google.antigravity.connections.local import test_utils

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
        compaction_config=types.CompactionConfig(max_context_tokens=12345),
    )
    self.assertEqual(config.compaction_config.max_context_tokens, 12345)
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._max_context_tokens, 12345)

  def test_litert_config_default_capabilities(self):
    """Verify LiteRTAgentConfig defaults to all capabilities enabled."""
    litert_config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
    )
    self.assertIsNone(litert_config.capabilities.enabled_tools)
    self.assertIsNone(litert_config.capabilities.disabled_tools)

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
    self.assertEqual(config.capabilities.compaction_threshold, 65536)
    self.assertFalse(config.capabilities.enable_subagents)

  def test_litert_config_lightweight_method_with_overrides(self):
    """Verify LiteRTAgentConfig.lightweight respects capability overrides."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/tmp/model.litertlm",
        backend=litert_connection_config.LiteRTBackend.CPU,
        capabilities=types.CapabilitiesConfig(compaction_threshold=3000),
    ).lightweight()
    self.assertIsInstance(config, litert_connection_config.LiteRTAgentConfig)
    self.assertEqual(config.capabilities.compaction_threshold, 3000)
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertFalse(config.capabilities.enable_subagents)

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
    with mock.patch.object(
        strategy, "_shutdown_server"
    ) as mock_shutdown, mock.patch.object(
        strategy, "_close_engine"
    ) as mock_close:
      with self.assertRaises(RuntimeError):
        await strategy.__aenter__()
      mock_shutdown.assert_called_once()
      mock_close.assert_called_once()

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
    """Verify max_num_tokens (not max_context_tokens) is passed to litert_lm.Engine."""
    config = litert_connection_config.LiteRTAgentConfig(
        model_path="/dummy/path.litertlm",
        compaction_config=types.CompactionConfig(max_context_tokens=4096),
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    mock_engine_cls = mock.MagicMock()
    mock_engine_inst = mock.MagicMock()
    mock_engine_cls.return_value = mock_engine_inst
    mock_engine_inst.__enter__.return_value = mock.MagicMock()

    with mock.patch.object(
        litert_connection, "litert_lm"
    ) as mock_lm, mock.patch.object(
        litert_connection.litert_server, "LiteRTOpenAIServer"
    ), mock.patch.object(
        litert_connection, "_urlopen_no_proxy"
    ) as mock_urlopen, mock.patch(
        "os.path.exists", return_value=True
    ), mock.patch.object(
        local_connection.LocalConnectionStrategy,
        "__aenter__",
        return_value=None,
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
      self.assertEqual(kwargs["max_num_tokens"], 4096)

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
        compaction_config=types.CompactionConfig(max_context_tokens=65536),
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

    with mock.patch.object(
        litert_connection, "litert_lm"
    ) as mock_lm, mock.patch.object(
        litert_connection.litert_server,
        "LiteRTOpenAIServer",
        return_value=fake_server_inst,
    ), mock.patch.object(
        litert_connection, "_urlopen_no_proxy"
    ) as mock_urlopen, mock.patch(
        "os.path.exists", return_value=True
    ), mock.patch(
        "google.antigravity.connections.local.local_connection.LocalConnectionStrategy.__aenter__",
        return_value=None,
    ):

      def fake_urlopen(req, timeout=None):
        url_str = str(
            req.full_url if hasattr(req, "full_url") else str(req)
        )
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


if __name__ == "__main__":
  unittest.main()
