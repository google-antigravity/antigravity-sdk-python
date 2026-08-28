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

"""Unit tests for LocalOpenAIConnectionStrategy and LocalOpenAIAgentConfig."""

import asyncio
import http.server
import json
import threading
import unittest
from unittest import mock

from google.antigravity.proto import localharness_pb2
from google.antigravity import agent
from google.antigravity import types
from google.antigravity.connections.local import local_connection
from google.antigravity.connections.local import local_openai_connection
from google.antigravity.connections.local import local_openai_connection_config
from google.antigravity.connections.local import test_utils
from google.antigravity.hooks import policy


class LocalOpenAIConnectionTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._real_binary_fn = local_connection._get_default_binary_path
    test_utils.patch_default_binary_path(self)

  def test_local_openai_strategy_harness_config(self):
    """Verify generic external OpenAI configuration works and clears Gemini config."""
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )

    self.assertIsInstance(
        strategy, local_openai_connection.LocalOpenAIConnectionStrategy
    )
    h_cfg = strategy._build_harness_config()

    # pylint: disable-next=g-generic-assert
    self.assertEqual(len(h_cfg.models), 1)
    model = h_cfg.models[0]
    self.assertEqual(model.name, "llama3.1")
    self.assertEqual(model.types, [localharness_pb2.MODEL_TYPE_TEXT])
    self.assertTrue(model.HasField("gemma_endpoint"))
    self.assertEqual(model.gemma_endpoint.base_url, "http://localhost:11434/v1")

  def test_local_openai_strategy_validate_empty_base_url(self):
    """Verify LocalOpenAIConnectionStrategy validates non-empty base_url."""
    strategy = local_openai_connection.LocalOpenAIConnectionStrategy(
        base_url="",
        model_name="test",
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    with self.assertRaises(types.AntigravityValidationError):
      strategy._validate_connection()

  def test_local_openai_config_model_target_parsing(self):
    """Verify LocalOpenAIAgentConfig parses model and endpoint base_url from ModelTarget."""
    endpoint = types.GeminiAPIEndpoint(base_url="http://custom-ollama:11434/v1")
    target = types.ModelTarget(name="llama3.2", endpoint=endpoint)
    config = local_openai_connection_config.LocalOpenAIAgentConfig(model=target)
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._model_name, "llama3.2")
    self.assertEqual(strategy._base_url, "http://custom-ollama:11434/v1")

  def test_local_openai_config_default_capabilities(self):
    """Verify LocalOpenAIAgentConfig defaults to all capabilities enabled."""
    openai_config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    )
    self.assertIsNone(openai_config.capabilities.enabled_tools)
    self.assertIsNone(openai_config.capabilities.disabled_tools)

  def test_local_openai_config_lightweight_method(self):
    """Verify LocalOpenAIAgentConfig.lightweight returns LocalOpenAIAgentConfig with defaults."""
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    ).lightweight()
    self.assertIsInstance(
        config, local_openai_connection_config.LocalOpenAIAgentConfig
    )
    self.assertEqual(config.base_url, "http://localhost:11434/v1")
    self.assertEqual(config.model, "llama3.1")
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertEqual(config.capabilities.compaction_threshold, 65536)
    self.assertFalse(config.capabilities.enable_subagents)

  def test_local_openai_config_lightweight_method_with_overrides(self):
    """Verify LocalOpenAIAgentConfig.lightweight respects capability overrides."""
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        capabilities=types.CapabilitiesConfig(compaction_threshold=4000),
    ).lightweight()
    self.assertIsInstance(
        config, local_openai_connection_config.LocalOpenAIAgentConfig
    )
    self.assertEqual(config.capabilities.compaction_threshold, 4000)
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertFalse(config.capabilities.enable_subagents)

  def test_local_openai_config_workspace_policies(self):
    """Verify LocalOpenAIAgentConfig does not prepend workspace_only policy."""
    config_openai = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost",
        model="m",
        workspaces=["/tmp/my_workspace"],
    )
    # LocalOpenAI uses localharness for native workspace containment;
    # config_openai.policies contains only the 2 confirm_run_command policies.
    # pylint: disable-next=g-generic-assert
    self.assertEqual(len(config_openai.policies), 2)
    self.assertEqual(config_openai.policies[0].tool, "run_command")
    self.assertEqual(config_openai.policies[1].tool, "*")

  def test_local_openai_config_mcp_servers_and_subagents_passed_to_strategy(
      self,
  ):
    """Verify LocalOpenAIAgentConfig passes mcp_servers and subagents to strategy."""
    mcp_server = types.McpStdioServer(
        name="test_mcp", command="echo", args=["hello"]
    )
    subagent = types.SubagentConfig(
        name="test_subagent",
        description="A test subagent",
        system_instructions="You are a subagent",
    )
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        mcp_servers=[mcp_server],
        subagents=[subagent],
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._mcp_servers, [mcp_server])
    self.assertEqual(strategy._subagents, [subagent])

  def test_local_openai_agent_getting_started_e2e(self):
    """End-to-end test executing Agent with LocalOpenAIAgentConfig and custom tools against a strict OpenAI mock server."""
    validation_errors = []
    received_tools = []

    class StrictOpenAIHandler(http.server.BaseHTTPRequestHandler):

      def do_POST(self):  # pylint: disable=invalid-name
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        body = json.loads(raw_body)

        if self.path == "/v1/chat/completions":
          tools = body.get("tools", [])
          received_tools.extend(tools)
          for i, tool in enumerate(tools):
            fn = tool.get("function", {})
            params = fn.get("parameters", {})
            schema_type = params.get("type")
            if schema_type == "OBJECT":
              validation_errors.append("Encountered uppercase 'OBJECT' type")
              err_resp = {
                  "error": {
                      "message": (
                          "Invalid discriminator value. Expected 'object',"
                          " got 'OBJECT'"
                      ),
                      "type": "invalid_request_error",
                      "param": f"tools[{i}].function.parameters.type",
                      "code": 400,
                  }
              }
              self.send_response(400)
              self.send_header("Content-Type", "application/json")
              self.end_headers()
              self.wfile.write(json.dumps(err_resp).encode("utf-8"))
              return

            props = params.get("properties", {})
            for prop_name, prop_schema in props.items():
              prop_type = prop_schema.get("type")
              if isinstance(prop_type, str) and prop_type.isupper():
                validation_errors.append(
                    f"Property '{prop_name}' has uppercase type '{prop_type}'"
                )
                err_resp = {
                    "error": {
                        "message": (
                            f"Invalid type '{prop_type}'. Expected"
                            f" '{prop_type.lower()}'"
                        ),
                        "type": "invalid_request_error",
                        "param": (
                            f"tools[{i}].function.parameters.properties.{prop_name}.type"
                        ),
                        "code": 400,
                    }
                }
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(err_resp).encode("utf-8"))
                return

          messages = body.get("messages", [])
          has_tool_response = any(m.get("role") == "tool" for m in messages)

          self.send_response(200)
          self.send_header("Content-Type", "text/event-stream")
          self.send_header("Cache-Control", "no-cache")
          self.end_headers()

          if not has_tool_response:
            chunk1 = {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": (
                                    json.dumps({"location": "Seattle"})
                                ),
                            },
                        }]
                    }
                }]
            }
            chunk2 = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
            self.wfile.write(f"data: {json.dumps(chunk1)}\n\n".encode("utf-8"))
            self.wfile.write(f"data: {json.dumps(chunk2)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
          else:
            chunk1 = {
                "choices": [{
                    "delta": {
                        "content": "The weather in Seattle is 72°F and sunny."
                    }
                }]
            }
            chunk2 = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(f"data: {json.dumps(chunk1)}\n\n".encode("utf-8"))
            self.wfile.write(f"data: {json.dumps(chunk2)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
        else:
          self.send_response(404)
          self.end_headers()

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    server = http.server.HTTPServer(("127.0.0.1", 0), StrictOpenAIHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    def get_current_weather(location: str) -> str:
      """Get the current weather for a given location."""
      return f"The weather in {location} is 72°F and sunny."

    async def run_agent():
      config = local_openai_connection_config.LocalOpenAIAgentConfig(
          base_url=f"http://127.0.0.1:{port}/v1",
          model="test-model",
          tools=[get_current_weather],
          policies=[policy.allow_all()],
      )
      with mock.patch.object(
          local_connection,
          "_get_default_binary_path",
          side_effect=self._real_binary_fn,
      ):
        async with agent.Agent(config) as ag:
          response = await ag.chat("What's the weather in Seattle?")
          tokens = [t async for t in response]
          return "".join(tokens)

    try:
      output = asyncio.run(run_agent())
      self.assertEqual(validation_errors, [])
      custom_tool = next(
          t for t in received_tools
          if t.get("function", {}).get("name") == "get_current_weather"
      )
      tool_params = custom_tool["function"]["parameters"]
      self.assertEqual(tool_params.get("type"), "object")
      self.assertEqual(
          tool_params.get("properties", {}).get("location", {}).get("type"),
          "string",
      )
      self.assertIn("Seattle", output)
    finally:
      server.shutdown()
      server.server_close()
      server_thread.join()

  def test_local_openai_agent_handles_http_400_error(self):
    """End-to-end test verifying that when an OpenAI server rejects requests with HTTP 400, the Agent session surfaces AntigravityExecutionError."""

    class RejectingOpenAIHandler(http.server.BaseHTTPRequestHandler):

      def do_POST(self):  # pylint: disable=invalid-name
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
          self.rfile.read(content_length)
        if self.path == "/v1/chat/completions":
          err_resp = {
              "error": {
                  "message": "Invalid request schema: Bad Request",
                  "type": "invalid_request_error",
                  "param": "tools[0].function.parameters.type",
                  "code": 400,
              }
          }
          self.send_response(400)
          self.send_header("Content-Type", "application/json")
          self.end_headers()
          self.wfile.write(json.dumps(err_resp).encode("utf-8"))
        else:
          self.send_response(404)
          self.end_headers()

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    server = http.server.HTTPServer(("127.0.0.1", 0), RejectingOpenAIHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    def get_current_weather(location: str) -> str:
      return f"The weather in {location} is 72°F and sunny."

    async def run_agent():
      config = local_openai_connection_config.LocalOpenAIAgentConfig(
          base_url=f"http://127.0.0.1:{port}/v1",
          model="test-model",
          tools=[get_current_weather],
          policies=[policy.allow_all()],
      )
      with mock.patch.object(
          local_connection,
          "_get_default_binary_path",
          side_effect=self._real_binary_fn,
      ):
        async with agent.Agent(config) as ag:
          response = await ag.chat("What's the weather in Seattle?")
          tokens = [t async for t in response]
          return "".join(tokens)

    try:
      with self.assertRaises(types.AntigravityExecutionError) as cm:
        asyncio.run(run_agent())
      self.assertIn(
          "Invalid request schema: Bad Request",
          str(cm.exception),
      )
    finally:
      server.shutdown()
      server.server_close()
      server_thread.join()


if __name__ == "__main__":
  unittest.main()
