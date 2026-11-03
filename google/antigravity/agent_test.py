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

"""Tests for Agent API."""

import contextlib
import errno
import os
import pathlib
import tempfile
from typing import Any, cast
import unittest
from unittest import mock

from google.antigravity import agent
from google.antigravity import types
from google.antigravity import workflows
from google.antigravity.connections import local as local_connection
from google.antigravity.connections.local import local_connection as lc_module
from google.antigravity.conversation import conversation
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy
from google.antigravity.models import DEFAULT_IMAGE_GENERATION_MODEL
from google.antigravity.models import DEFAULT_MODEL


class AgentTest(unittest.IsolatedAsyncioTestCase):

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_lifecycle(self, mock_conv_create, mock_strategy_class):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      self.assertEqual(ag._conversation, mock_conversation)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_chat(self, mock_conv_create, mock_strategy_class):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_response = mock.MagicMock(spec=types.ChatResponse)
    mock_response.text = mock.AsyncMock(return_value="Hello back")
    mock_conversation.chat = mock.AsyncMock(return_value=mock_response)

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      response = await ag.chat("Hello")
      self.assertEqual(await response.text(), "Hello back")
      mock_conversation.chat.assert_called_once_with("Hello")

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_agent_chat_multimodal_input(
      self, mock_conv_create, mock_strategy_class
  ):
    """Verifies that the Agent public API method accepts multimodal Content payloads."""
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_response = mock.MagicMock(spec=types.ChatResponse)
    mock_response.text = mock.AsyncMock(return_value="Analyzed image content")
    mock_conversation.chat = mock.AsyncMock(return_value=mock_response)

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      multimodal_prompt = [
          "Look at this:",
          types.Image(mime_type="image/png", data=b"png_bytes"),
      ]
      response = await ag.chat(multimodal_prompt)
      self.assertEqual(await response.text(), "Analyzed image content")
      mock_conversation.chat.assert_called_once_with(multimodal_prompt)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_chat_empty_input_raises_value_error(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      with self.assertRaisesRegex(ValueError, "non-empty message string"):
        await ag.chat(None)
      with self.assertRaisesRegex(ValueError, "non-empty message string"):
        await ag.chat("")
      with self.assertRaisesRegex(ValueError, "non-empty message string"):
        await ag.chat("   ")
      with self.assertRaisesRegex(ValueError, "non-empty message content"):
        await ag.chat([])
      with self.assertRaisesRegex(ValueError, "non-empty message content"):
        await ag.chat(["", "   "])
      with self.assertRaisesRegex(ValueError, "non-empty message content"):
        await ag.chat(())
      with self.assertRaisesRegex(ValueError, "non-empty message content"):
        await ag.chat(("", "   "))
      mock_conversation.chat.assert_not_called()

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_collects_subagent_custom_tools_into_tool_runner(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    def root_tool():
      """Root tool."""

    def sub_tool():
      """Subagent tool."""

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        tools=[root_tool],
        subagents=[
            types.SubagentConfig(
                name="helper",
                description="helper",
                tools=[sub_tool],
            )
        ],
    )
    async with agent.Agent(config) as ag:
      self.assertIsNotNone(ag._tool_runner)
      self.assertIn("root_tool", ag._tool_runner.tools)
      self.assertIn("sub_tool", ag._tool_runner.tools)

  def test_agent_config_get_all_custom_tools_guards_none_and_deduplicates(self):
    # Test None guards return empty list
    empty_config = local_connection.LocalAgentConfig(tools=None, subagents=None)
    self.assertEqual(empty_config._get_all_custom_tools(), [])

    # Test subagents with None tools returns empty list
    sub_none_tools_config = local_connection.LocalAgentConfig(
        tools=None,
        subagents=[
            types.SubagentConfig(name="sub", description="sub", tools=None)
        ],
    )
    self.assertEqual(sub_none_tools_config._get_all_custom_tools(), [])

    # Test filtering out string tools and preserving callables
    def shared_tool():
      """Shared tool."""

    def sub_only_tool():
      """Sub only tool."""

    config = local_connection.LocalAgentConfig(
        tools=[shared_tool, "view_file"],
        subagents=[
            types.SubagentConfig(
                name="sub1",
                description="sub1",
                tools=[shared_tool, sub_only_tool, "grep_search"],
            ),
            types.SubagentConfig(
                name="sub2",
                description="sub2",
                tools=[sub_only_tool],
            ),
        ],
    )
    all_tools = config._get_all_custom_tools()
    self.assertEqual(len(all_tools), 2)
    self.assertIn(shared_tool, all_tools)
    self.assertIn(sub_only_tool, all_tools)

  def test_agent_config_get_all_custom_tools_raises_on_name_collision(self):
    def tool_fn():
      """Root tool."""

    # Define a distinct function object with the exact same name
    def make_conflicting_tool():
      def tool_fn():
        """Conflicting sub tool."""

      return tool_fn

    conflicting_tool = make_conflicting_tool()

    config = local_connection.LocalAgentConfig(
        tools=[tool_fn],
        subagents=[
            types.SubagentConfig(
                name="sub1",
                description="sub1",
                tools=[conflicting_tool],
            ),
        ],
    )
    with self.assertRaisesRegex(
        ValueError,
        "Duplicate custom tool name 'tool_fn' detected across agent and"
        " subagent 'sub1' configurations.",
    ):
      config._get_all_custom_tools()

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_default_capabilities(
      self, mock_conv_create, mock_strategy_class
  ):
    """Default LocalAgentConfig enables all tools (permissive)."""
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertIsNone(capabilities_config.enabled_tools)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_requires_policies_in_write_mode(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_explicit_write_tool(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when enabled_tools includes a non-read-only tool."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=[types.BuiltinTools.RUN_COMMAND],
        ),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_explicit_all_tools(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when all tools are listed explicitly."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=list(types.BuiltinTools),
        ),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_empty_disabled_tools(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when disabled_tools=[] (= all tools enabled)."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(disabled_tools=[]),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_read_only_explicit_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when only read-only tools are explicitly enabled."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=types.BuiltinTools.read_only(),
        ),
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_deprecated_read_only_explicit_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when deprecated read-only tools are explicitly enabled."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=types.BuiltinTools.deprecated(),
        ),
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_disabling_all_default_write_tools_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """Disabling all default write tools leaves only read-only tools active."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    default_write_tools = list(
        set(types.BuiltinTools.default())
        - set(types.BuiltinTools.read_only())
    )
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            disabled_tools=default_write_tools,
        ),
        policies=[],
        workspaces=[],
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_write_tools_with_policy_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when write tools are enabled AND policies are provided."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[policy.deny("*")],
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_mcp_server_requires_policies_or_hook(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when MCP servers are present but no policies or hooks."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        mcp_servers=cast(
            Any,
            [{
                "type": "stdio",
                "name": "test_server",
                "command": "node",
                "args": ["index.js"],
            }],
        ),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_mcp_server_with_policy_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when MCP servers are present AND policies are provided."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        mcp_servers=cast(
            Any,
            [{
                "type": "stdio",
                "name": "test_server",
                "command": "node",
                "args": ["index.js"],
            }],
        ),
        policies=[policy.deny("*")],
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_mcp_server_with_hook_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when MCP servers are present AND a decide hook is provided."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())

    class MyPreToolCallDecideHook(hooks.PreToolCallDecideHook):

      async def run(self, context, data):
        return types.HookResult(allow=True)

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        mcp_servers=cast(
            Any,
            [{
                "type": "stdio",
                "name": "test_server",
                "command": "node",
                "args": ["index.js"],
            }],
        ),
        hooks=[MyPreToolCallDecideHook()],
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_register_hook(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    class MyPreTurnHook(hooks.PreTurnHook):

      async def run(self, context, data):
        return types.HookResult(allow=True)

    my_hook = MyPreTurnHook()

    # Test constructor registration
    config = local_connection.LocalAgentConfig(
        system_instructions="test", hooks=[my_hook]
    )
    async with agent.Agent(config) as ag:
      self.assertIsNotNone(ag._hook_runner)
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch(
      "google.antigravity.agent."
      "trigger_runner.TriggerRunner"
  )
  async def test_agent_register_trigger(
      self,
      mock_trigger_runner_class,
      mock_conv_create,
      mock_strategy_class,
  ):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()

    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_runner_instance = mock.AsyncMock()
    mock_trigger_runner_class.return_value = mock_runner_instance

    async def _simulate_aenter(*args, **kwargs):
      await mock_runner_instance.start()
      return mock_runner_instance

    mock_runner_instance.__aenter__ = mock.AsyncMock(
        side_effect=_simulate_aenter
    )

    async def _simulate_aexit(*args):
      await mock_runner_instance.stop()

    mock_runner_instance.__aexit__ = mock.AsyncMock(side_effect=_simulate_aexit)

    async def my_trigger(ctx):
      del ctx  # Unused.
      pass

    # Test constructor registration: TriggerRunner started with trigger.
    config = local_connection.LocalAgentConfig(
        system_instructions="test", triggers=[my_trigger]
    )
    async with agent.Agent(config):
      mock_trigger_runner_class.assert_called_once()
      call_kwargs = mock_trigger_runner_class.call_args[1]
      self.assertEqual(call_kwargs["triggers"], [my_trigger])
      mock_runner_instance.start.assert_called_once()

    # TriggerRunner.stop() called during __aexit__.
    mock_runner_instance.stop.assert_called_once()

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_write_mode_with_policies(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    my_policy = policy.allow("some_tool")

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[my_policy],
    )
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertNotEqual(
          capabilities_config.enabled_tools, types.BuiltinTools.read_only()
      )

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_mcp_server_unknown_type(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mcp_servers = cast(Any, [{"type": "unknown_type"}])

    with self.assertRaises(ValueError):
      config = local_connection.LocalAgentConfig(
          system_instructions="test", mcp_servers=mcp_servers
      )
      async with agent.Agent(config):
        pass

  async def test_agent_chat_before_start(self):
    ag = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    with self.assertRaises(RuntimeError):
      await ag.chat("hello")

  async def test_agent_conversation_before_start(self):
    """Verifies conversation raises RuntimeError before session starts."""
    ag = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    with self.assertRaises(RuntimeError):
      _ = ag.conversation

  async def test_agent_is_started_before_start(self):
    """Verifies is_started returns False before session starts."""
    ag = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    self.assertFalse(ag.is_started)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_is_started_after_start(
      self, mock_conv_create, mock_strategy_class
  ):
    """Verifies is_started returns True inside async with."""
    del mock_conv_create  # Unused.
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      self.assertTrue(ag.is_started)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_api_key_env(self, mock_conv_create, mock_strategy_class):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    with mock.patch.dict("os.environ", {}, clear=True):
      config = local_connection.LocalAgentConfig(
          system_instructions="test", api_key="test_key"
      )
      async with agent.Agent(config):
        self.assertIsNone(os.environ.get("GEMINI_API_KEY"))
        _, kwargs = mock_strategy_class.call_args
        models = kwargs.get("models")
        self.assertIsNotNone(models)
        text_model = next(
            m for m in models if types.ModelType.TEXT in m.types
        )
        self.assertEqual(text_model.endpoint.api_key, "test_key")

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_model_sugar_flows_to_strategy(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(
        system_instructions="test", model="gemini-2.5-pro"
    )
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      models = kwargs.get("models")
      self.assertIsNotNone(models)
      text_model = next(
          m for m in models if types.ModelType.TEXT in m.types
      )
      self.assertEqual(text_model.name, "gemini-2.5-pro")

  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_system_instructions_object(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    si_obj = types.CustomSystemInstructions(text="custom si")
    config = local_connection.LocalAgentConfig(system_instructions=si_obj)
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      si = kwargs.get("system_instructions")
      self.assertEqual(si, si_obj)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_session_config(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        conversation_id="12345678901234567890123456789012",
        save_dir="/state",
        workspaces=["/path/1", "/path/2"],
    )
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      self.assertEqual(
          kwargs.get("conversation_id"), "12345678901234567890123456789012"
      )
      self.assertEqual(kwargs.get("save_dir"), "/state")
      self.assertEqual(kwargs.get("workspaces"), ["/path/1", "/path/2"])

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_skills_paths(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    skills_paths = ["/path/1", "/path/2"]
    config = local_connection.LocalAgentConfig(
        system_instructions="test", skills_paths=skills_paths
    )
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      sp = kwargs.get("skills_paths")
      self.assertEqual(sp, skills_paths)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_inline_skills(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    inline_skills = [
        types.InlineSkill(
            name="my-skill",
            description="A test skill.",
            content="# Instructions",
        )
    ]
    config = local_connection.LocalAgentConfig(
        system_instructions="test", inline_skills=inline_skills
    )
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      self.assertEqual(kwargs.get("inline_skills"), inline_skills)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_conversation_after_start(
      self, mock_conv_create, mock_strategy_class
  ):
    """Verifies conversation returns the Conversation instance after start."""
    mock_strategy_instance = mock.MagicMock()

    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation.connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      self.assertEqual(ag.conversation, mock_conversation)

  async def test_agent_aexit_passes_exceptions(self):
    config = local_connection.LocalAgentConfig(system_instructions="test")
    ag = agent.Agent(config)

    mock_exit_stack = mock.AsyncMock(spec=contextlib.AsyncExitStack)
    ag._exit_stack = mock_exit_stack

    # Simulate entering the context
    ag._conversation = mock.MagicMock()

    exc = ValueError("test exception")
    await ag.__aexit__(ValueError, exc, None)

    mock_exit_stack.__aexit__.assert_called_once_with(ValueError, exc, None)

  async def test_agent_aexit_returns_suppression_status(self):
    config = local_connection.LocalAgentConfig(system_instructions="test")
    ag = agent.Agent(config)

    mock_exit_stack = mock.AsyncMock(spec=contextlib.AsyncExitStack)
    mock_exit_stack.__aexit__.return_value = True
    ag._exit_stack = mock_exit_stack

    # Simulate entering the context
    ag._conversation = mock.MagicMock()

    exc = ValueError("test exception")
    suppressed = await ag.__aexit__(ValueError, exc, None)

    self.assertTrue(suppressed)
    mock_exit_stack.__aexit__.assert_called_once_with(ValueError, exc, None)


class AgentConfigTest(unittest.IsolatedAsyncioTestCase):
  """Tests for AgentConfig sugar, conflict guards, and defensive copy."""

  def test_sugar_model_flows_to_models(self):
    """Verifies model sugar flows to config.models."""
    config = local_connection.LocalAgentConfig(
        system_instructions="test", model="gemini-2.5-pro"
    )
    self.assertIsNotNone(config.models)
    text_model = [m for m in config.models if types.ModelType.TEXT in m.types][
        0
    ]
    self.assertEqual(text_model.name, "gemini-2.5-pro")

  def test_sugar_api_key_flows_to_models(self):
    """Verifies api_key sugar flows to config.models."""
    config = local_connection.LocalAgentConfig(
        system_instructions="test", api_key="my-key"
    )
    self.assertIsNotNone(config.models)
    text_model = [m for m in config.models if types.ModelType.TEXT in m.types][
        0
    ]
    self.assertIsInstance(text_model.endpoint, types.GeminiAPIEndpoint)
    self.assertEqual(text_model.endpoint.api_key, "my-key")

  def test_defaults(self):
    """Verifies AgentConfig defaults: safe policies, default model."""
    config = local_connection.LocalAgentConfig(system_instructions="test")
    self.assertIsNone(config.capabilities.enabled_tools)
    self.assertIsNone(config.capabilities.disabled_tools)
    # Default includes only the 2 policies from policy.confirm_run_command();
    # workspace containment is enforced natively by localharness.
    self.assertEqual(len(config.policies), 2)
    self.assertEqual(config.policies[0].tool, "run_command")
    self.assertEqual(config.policies[1].tool, "*")
    self.assertIsNotNone(config.models)
    text_model = [m for m in config.models if types.ModelType.TEXT in m.types][
        0
    ]
    self.assertEqual(text_model.name, DEFAULT_MODEL)

  def test_model_sugar_does_not_clobber_image_generation(self):
    """Verifies model sugar only sets default slot, not image_generation."""
    config = local_connection.LocalAgentConfig(
        system_instructions="test", model="custom-chat-model"
    )
    self.assertIsNotNone(config.models)
    text_model = [m for m in config.models if types.ModelType.TEXT in m.types][
        0
    ]
    image_model = [
        m for m in config.models if types.ModelType.IMAGE in m.types
    ][0]
    self.assertEqual(text_model.name, "custom-chat-model")
    self.assertEqual(image_model.name, DEFAULT_IMAGE_GENERATION_MODEL)

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_agent_with_response_schema(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    schema_dict = {"properties": {"field": {"type": "string"}}}
    config = local_connection.LocalAgentConfig(
        system_instructions="test", response_schema=schema_dict
    )
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertEqual(
          capabilities_config.finish_tool_schema_json,
          '{"properties": {"field": {"type": "string"}}}',
      )

  def test_conversation_id_returns_none_before_session(self):
    """Verifies conversation_id is None before the session starts."""
    a = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    self.assertIsNone(a.conversation_id)

  def test_conversation_id_returns_value_after_session(self):
    """Verifies conversation_id returns the runtime-assigned ID."""
    a = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    mock_conv = mock.MagicMock()
    mock_conv.conversation_id = "test-conv-123"
    a._conversation = mock_conv
    self.assertEqual(a.conversation_id, "test-conv-123")

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_auto_policy_allows_write_tools(
      self, mock_conv_create, mock_strategy_class
  ):
    """When policy.auto() is present, write tools should not be restricted."""
    del mock_conv_create  # Unused.
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[policy.auto()],
    )
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      active_tools = agent.connection_module.resolve_active_tools(
          capabilities_config
      )
      self.assertIn(types.BuiltinTools.RUN_COMMAND, active_tools)
      policies = kwargs.get("policies")
      self.assertTrue(any(isinstance(p, policy.AutoPolicy) for p in policies))

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_run_workflow_with_decorated_function(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_conversation.history = []
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    @workflows.define
    def sample_workflow():
      """Sample workflow docstring."""
      workflows.phase("Explore")
      res = workflows.agent("Check files")
      workflows.log("Result:", res)

    captured_script_paths: list[str] = []

    async def fake_chat(prompt: str):
      # Extract script path from prompt and verify it exists during execution.
      for token in prompt.split("`"):
        if token.endswith(".py"):
          captured_script_paths.append(token)
          self.assertTrue(pathlib.Path(token).is_file())
          break
      mock_conversation.history.append(
          types.Step(
              id="traj-1:1",
              step_index=1,
              type=types.StepType.TOOL_CALL,
              status=types.StepStatus.DONE,
              tool_calls=[
                  types.ToolCall(
                      name="run_workflow",
                      args={
                          "script_path": captured_script_paths[0],
                          "description": "Sample workflow docstring.",
                      },
                  )
              ],
              workflow_progress=types.WorkflowProgress(
                  script_path=captured_script_paths[0],
                  description="Sample workflow docstring.",
                  output="Result: ok",
              ),
          )
      )
      resp = mock.MagicMock(spec=types.ChatResponse)
      resp.text = mock.AsyncMock(return_value="Workflow completed.")
      return resp

    mock_conversation.chat = mock.AsyncMock(side_effect=fake_chat)

    with tempfile.TemporaryDirectory() as ws_dir:
      config = local_connection.LocalAgentConfig(
          system_instructions="test",
          workspaces=[ws_dir],
      )
      async with agent.Agent(config) as ag:
        self.assertFalse(hasattr(ag, "run_workflow"))
        result = await ag.beta.run_workflow(sample_workflow)
        self.assertIsInstance(result, types.ChatResponse)
        self.assertEqual(result.description, "Sample workflow docstring.")
        self.assertEqual(result.output, "Result: ok")
        self.assertEqual(result.response_text, "Workflow completed.")
        self.assertEqual(len(captured_script_paths), 1)
        # Temp script must be cleaned up after run_workflow completes.
        self.assertFalse(pathlib.Path(captured_script_paths[0]).exists())

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_run_workflow_with_undecorated_function_and_chat_response_accessors(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_conversation.history = []
    mock_conversation.get_last_structured_output.return_value = {"status": "ok"}
    mock_conversation.last_turn_usage = types.UsageMetadata(
        prompt_token_count=12,
        candidates_token_count=8,
        total_token_count=20,
    )
    mock_conversation._last_turn_stop_reason = types.StopReason.UNSPECIFIED
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    async def raw_workflow():
      """Undecorated workflow docstring."""
      workflows.phase("Inspect")
      workflows.log("inspected")

    captured_prompts: list[str] = []

    async def fake_chat(prompt: str):
      captured_prompts.append(prompt)
      mock_conversation.history.append(
          types.Step(
              id="traj-1:1",
              step_index=1,
              type=types.StepType.TOOL_CALL,
              status=types.StepStatus.DONE,
              tool_calls=[
                  types.ToolCall(
                      name="run_workflow",
                      args={"description": "Undecorated workflow docstring."},
                  )
              ],
              workflow_progress=types.WorkflowProgress(
                  output="inspected",
              ),
          )
      )

      async def _stream():
        yield types.Thought(step_index=1, text="Planning workflow.")
        yield types.Text(step_index=2, text="Workflow completed.")

      return types.ChatResponse(_stream(), conversation=mock_conversation)

    mock_conversation.chat = mock.AsyncMock(side_effect=fake_chat)

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      result = await ag.beta.run_workflow(raw_workflow)
      self.assertIsInstance(result, types.ChatResponse)
      self.assertEqual(len(captured_prompts), 1)
      self.assertIn("Undecorated workflow docstring.", captured_prompts[0])
      self.assertEqual(result.description, "Undecorated workflow docstring.")
      self.assertEqual(result.output, "inspected")
      self.assertEqual(result.response_text, "Workflow completed.")
      self.assertEqual(await result.text(), "Workflow completed.")
      self.assertEqual(await result.structured_output(), {"status": "ok"})
      self.assertEqual(
          [t async for t in result.thoughts], ["Planning workflow."]
      )
      self.assertIsNotNone(result.usage_metadata)
      self.assertEqual(result.usage_metadata.total_token_count, 20)
      self.assertEqual(result.stop_reason, types.StopReason.UNSPECIFIED)

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_run_workflow_raises_when_disabled(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    @workflows.define
    def noop_wf():
      workflows.phase("Noop")

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            disabled_tools=[types.BuiltinTools.RUN_WORKFLOW]
        ),
    )
    async with agent.Agent(config) as ag:
      with self.assertRaisesRegex(
          ValueError, "BuiltinTools.RUN_WORKFLOW is not enabled"
      ):
        await ag.beta.run_workflow(noop_wf)

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_run_workflow_raises_on_step_error(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_conversation.history = []
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    @workflows.define
    def failing_wf():
      workflows.phase("Fail")

    async def fake_chat(prompt: str):
      del prompt
      mock_conversation.history.append(
          types.Step(
              id="traj-1:1",
              step_index=1,
              type=types.StepType.TOOL_CALL,
              status=types.StepStatus.ERROR,
              error="workflow execution failed: RuntimeError: boom",
              tool_calls=[types.ToolCall(name="run_workflow", args={})],
          )
      )
      resp = mock.MagicMock(spec=types.ChatResponse)
      resp.text = mock.AsyncMock(return_value="Failed.")
      return resp

    mock_conversation.chat = mock.AsyncMock(side_effect=fake_chat)

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      with self.assertRaisesRegex(
          types.ToolExecutionError, "workflow execution failed"
      ) as cm:
        await ag.beta.run_workflow(failing_wf)
      self.assertEqual(
          cm.exception.tool_name, types.BuiltinTools.RUN_WORKFLOW.value
      )

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_run_workflow_with_script_path_preserves_user_file(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_conversation.history = []
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    with tempfile.TemporaryDirectory() as ws_dir:
      script_file = pathlib.Path(ws_dir) / "my_workflow.py"
      script_file.write_text("phase('Audit')\nlog('done')\n", encoding="utf-8")

      captured_prompts = []

      async def fake_chat(prompt: str):
        captured_prompts.append(prompt)
        self.assertIn(str(script_file.resolve()), prompt)
        mock_conversation.history.append(
            types.Step(
                id="traj-1:1",
                step_index=1,
                type=types.StepType.TOOL_CALL,
                status=types.StepStatus.DONE,
                tool_calls=[
                    types.ToolCall(
                        name="run_workflow",
                        args={"script_path": str(script_file.resolve())},
                    )
                ],
                workflow_progress=types.WorkflowProgress(
                    script_path=str(script_file.resolve()),
                    description="Run workflow my_workflow.py",
                    output="done",
                ),
            )
        )
        resp = mock.MagicMock(spec=types.ChatResponse)
        resp.text = mock.AsyncMock(return_value="Completed.")
        return resp

      mock_conversation.chat = mock.AsyncMock(side_effect=fake_chat)

      config = local_connection.LocalAgentConfig(
          system_instructions="test",
          workspaces=[ws_dir],
      )
      async with agent.Agent(config) as ag:
        # Pass relative script path resolved against workspace and workflow
        # args.
        res = await ag.beta.run_workflow(
            script_path="my_workflow.py",
            args={"target": "auth.py", "retries": 2},
        )
        self.assertEqual(res.script_path, str(script_file.resolve()))
        self.assertEqual(res.output, "done")
        self.assertIn(
            'with Args `{"target": "auth.py", "retries": 2}`',
            captured_prompts[-1],
        )
        await ag.beta.run_workflow(script_path="my_workflow.py", args={})
        self.assertNotIn("with Args", captured_prompts[-1])
        # User's script file must NOT be deleted after run_workflow completes.
        self.assertTrue(script_file.is_file())

        bad_args: Any = ["not", "a", "dict"]
        with self.assertRaisesRegex(
            TypeError, r"run_workflow\(\) `args` must be a mapping"
        ):
          await ag.beta.run_workflow(
              script_path="my_workflow.py",
              args=bad_args,
          )

        with self.assertRaisesRegex(
            ValueError, r"run_workflow\(\) `args` must be JSON-serializable"
        ):
          await ag.beta.run_workflow(
              script_path="my_workflow.py",
              args={"bad": object()},
          )

        with self.assertRaisesRegex(
            ValueError, r"run_workflow\(\) `args` must be JSON-serializable"
        ):
          await ag.beta.run_workflow(
              script_path="my_workflow.py",
              args={"bad_float": float("nan")},
          )

        with self.assertRaises(FileNotFoundError):
          await ag.beta.run_workflow(script_path="does_not_exist.py")

        # Passing neither or both of `workflow` and `script_path` must fail.
        with self.assertRaisesRegex(
            ValueError,
            "requires exactly one of `workflow` or `script_path`",
        ):
          await ag.beta.run_workflow()

        with self.assertRaisesRegex(
            ValueError,
            "requires exactly one of `workflow` or `script_path`",
        ):
          await ag.beta.run_workflow(
              lambda: None, script_path=str(script_file.resolve())
          )

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_run_workflow_raises_when_tool_not_invoked(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_conversation.history = []
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    @workflows.define
    def sample_wf():
      workflows.phase("Check")

    async def fake_chat(prompt: str):
      del prompt
      resp = mock.MagicMock(spec=types.ChatResponse)
      resp.text = mock.AsyncMock(return_value="I did not call run_workflow.")
      return resp

    mock_conversation.chat = mock.AsyncMock(side_effect=fake_chat)

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      with self.assertRaisesRegex(
          types.ToolExecutionError,
          "The model did not invoke the run_workflow tool",
      ) as cm:
        await ag.beta.run_workflow(sample_wf)
      self.assertEqual(
          cm.exception.tool_name, types.BuiltinTools.RUN_WORKFLOW.value
      )

  def test_materialize_workflow_function_falls_back_across_workspace_errors(
      self,
  ):
    @workflows.define
    def ro_wf():
      workflows.phase("ReadOnlyCheck")

    real_ntf = tempfile.NamedTemporaryFile
    call_dirs = []

    def fake_ntf(*args, **kwargs):
      call_dirs.append(kwargs.get("dir"))
      if kwargs.get("dir") is not None:
        raise OSError(errno.EROFS, "Read-only file system")
      return real_ntf(*args, **kwargs)

    with tempfile.TemporaryDirectory() as ws_dir:
      resolved_ws = str(pathlib.Path(ws_dir).resolve())
      with mock.patch.object(
          tempfile, "NamedTemporaryFile", side_effect=fake_ntf
      ):
        path, _, _ = agent._materialize_workflow_function(
            ro_wf, [ws_dir], "desc"
        )
      try:
        self.assertEqual(call_dirs, [resolved_ws, None])
        self.assertTrue(os.path.isfile(path))
      finally:
        if os.path.exists(path):
          os.unlink(path)

    # Also verify fallback from a failing first workspace to a writable second
    # workspace.
    with (
        tempfile.TemporaryDirectory() as ws_ro,
        tempfile.TemporaryDirectory() as ws_rw,
    ):
      resolved_ro = str(pathlib.Path(ws_ro).resolve())
      resolved_rw = str(pathlib.Path(ws_rw).resolve())

      def fake_ntf_second_writable(*args, **kwargs):
        if kwargs.get("dir") == resolved_ro:
          raise RuntimeError("Unusable workspace mount")
        return real_ntf(*args, **kwargs)

      with mock.patch.object(
          tempfile, "NamedTemporaryFile", side_effect=fake_ntf_second_writable
      ):
        path2, _, _ = agent._materialize_workflow_function(
            ro_wf, [ws_ro, ws_rw], "desc"
        )
      try:
        self.assertTrue(path2.startswith(resolved_rw))
        self.assertTrue(os.path.isfile(path2))
      finally:
        if os.path.exists(path2):
          os.unlink(path2)

    # Also verify partial temp file cleanup when write() fails after file
    # creation.
    with (
        tempfile.TemporaryDirectory() as ws_full,
        tempfile.TemporaryDirectory() as ws_ok,
    ):
      resolved_full = str(pathlib.Path(ws_full).resolve())
      resolved_ok = str(pathlib.Path(ws_ok).resolve())
      leaked_paths = []

      def fake_ntf_write_error(*args, **kwargs):
        f = real_ntf(*args, **kwargs)
        if kwargs.get("dir") == resolved_full:
          leaked_paths.append(f.name)
          f.write = mock.MagicMock(
              side_effect=OSError(errno.ENOSPC, "No space left on device")
          )
        return f

      with mock.patch.object(
          tempfile, "NamedTemporaryFile", side_effect=fake_ntf_write_error
      ):
        path3, _, _ = agent._materialize_workflow_function(
            ro_wf, [ws_full, ws_ok], "desc"
        )
      try:
        self.assertEqual(len(leaked_paths), 1)
        self.assertFalse(os.path.exists(leaked_paths[0]))
        self.assertTrue(path3.startswith(resolved_ok))
        self.assertTrue(os.path.isfile(path3))
      finally:
        if os.path.exists(path3):
          os.unlink(path3)

  def test_resolve_workflow_script_path_skips_inaccessible_workspaces(self):
    with tempfile.TemporaryDirectory() as ws_dir:
      script_file = pathlib.Path(ws_dir) / "wf.py"
      script_file.write_text("phase('Run')\n", encoding="utf-8")
      resolved, desc = agent._resolve_workflow_script_path(
          "wf.py",
          ["\x00invalid_ws", "/nonexistent/unmounted/workspace", ws_dir],
          "",
      )
      self.assertEqual(resolved, str(script_file.resolve()))
      self.assertEqual(desc, "Run workflow wf.py")

      with self.assertRaises(FileNotFoundError):
        agent._resolve_workflow_script_path(
            "\x00invalid_script.py", [ws_dir], ""
        )

    for bad_source in ("phase('a')\x00", "break\n"):
      with self.assertRaisesRegex(
          workflows.WorkflowError, "workflow syntax error"
      ):
        workflows.validate_workflow_source(bad_source)

    with mock.patch.object(
        workflows.inspect, "getsource", side_effect=OSError("no source")
    ):
      with self.assertRaisesRegex(
          workflows.WorkflowError, "cannot inspect source of workflow function"
      ):
        workflows.extract_workflow_source(lambda: None)

    with mock.patch.object(
        workflows.inspect, "getsource", return_value="def wf():\n  \x00\n"
    ):
      with self.assertRaisesRegex(
          workflows.WorkflowError, "workflow syntax error"
      ):
        workflows.extract_workflow_source(lambda: None)


if __name__ == "__main__":
  unittest.main()
