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

"""Validates default implementations in the Connection abstract base class."""

import ntpath
import os
import pathlib
from typing import Any, cast
import unittest
from unittest import mock
import warnings
import pydantic
from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.hooks import hooks as hooks_mod
from google.antigravity.hooks import policy


class DummyConnection(connection.Connection):

  async def send(self, prompt: str, **kwargs) -> None:
    pass

  def receive_steps(self):
    pass

  async def disconnect(self) -> None:
    pass

  async def send_trigger_notification(self, content: str) -> None:
    pass


class ConnectionTest(unittest.IsolatedAsyncioTestCase):

  async def test_default_implementations(self):
    conn = DummyConnection()

    self.assertTrue(conn.is_idle)
    self.assertEqual(conn.conversation_id, "")

    await conn.cancel()
    await conn.wait_for_idle()
    self.assertFalse(await conn.wait_for_wakeup())
    await conn._send_tool_results([])
    self.assertIsNone(conn.debug_config)


class DebugConfigTest(unittest.TestCase):

  def test_debug_config_defaults(self):
    cfg = connection.DebugConfig()
    self.assertTrue(cfg.enable_server_side_tracing)
    self.assertIsNotNone(cfg.logging_level)

  def test_agent_config_debug_validation(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    cfg_bool = ConcreteConfig(debug_config=True)
    self.assertIsInstance(cfg_bool.debug_config, connection.DebugConfig)
    self.assertTrue(cfg_bool.debug_config.enable_server_side_tracing)

    cfg_false = ConcreteConfig(debug_config=False)
    self.assertIsNone(cfg_false.debug_config)

    cfg_dict = ConcreteConfig(
        debug_config={"enable_server_side_tracing": False}
    )
    self.assertIsInstance(cfg_dict.debug_config, connection.DebugConfig)
    self.assertFalse(cfg_dict.debug_config.enable_server_side_tracing)


class AgentConfigTest(unittest.TestCase):

  def test_cannot_instantiate_abc(self):
    with self.assertRaises(TypeError):
      connection.AgentConfig(system_instructions="test")

  def test_subclass_must_implement_create_strategy(self):
    class IncompleteConfig(connection.AgentConfig):
      pass

    with self.assertRaises(TypeError):
      IncompleteConfig(system_instructions="test")

  def test_concrete_subclass_works(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(system_instructions="test")
    self.assertEqual(config.system_instructions, "test")
    self.assertEqual(config.inline_skills, [])

    skill = types.InlineSkill(name="s1", description="d", content="c")
    config_with_skills = ConcreteConfig(inline_skills=[skill])
    self.assertEqual(config_with_skills.inline_skills, [skill])

  def test_response_schema_valid_json_string(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(response_schema='{"type": "object"}')
    self.assertEqual(config.response_schema, '{"type": "object"}')

  def test_response_schema_invalid_json_raises(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaises(ValueError):
      ConcreteConfig(response_schema="not valid json {{{")

  def test_response_schema_unsupported_type_raises(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaises(ValueError):
      ConcreteConfig(response_schema=42)

  def test_policies_validation(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    my_policy = policy.Policy(tool="test", decision=policy.Decision.APPROVE)

    # Test policies=None (should default to empty list)
    config_none = ConcreteConfig(policies=None)
    self.assertEqual(config_none.policies, [])

    # Test policies as a single object (should be wrapped in a list)
    config_single = ConcreteConfig(policies=my_policy)
    self.assertEqual(config_single.policies, [my_policy])

    # Test policies as a nested list (should be flattened)
    config_nested = ConcreteConfig(policies=[my_policy, [my_policy]])
    self.assertEqual(config_nested.policies, [my_policy, my_policy])

  def test_model_copy_deep_preserves_executable_references(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    def my_tool():
      pass

    class DummyHook(hooks_mod.InspectHook):

      async def run(self, context, data):
        pass

    my_hook = DummyHook()

    async def my_trigger(_):
      pass

    my_policy = policy.Policy(tool="test", decision=policy.Decision.APPROVE)

    config = ConcreteConfig(
        tools=[my_tool],
        hooks=[my_hook],
        triggers=[my_trigger],
        policies=[my_policy],
    )
    copied = config.model_copy(deep=True)
    self.assertIs(copied.tools[0], my_tool)
    self.assertIs(copied.hooks[0], my_hook)
    self.assertIs(copied.triggers[0], my_trigger)
    self.assertIs(copied.policies[0], my_policy)

  def test_session_continuation_validation_success(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    # Should not raise
    ConcreteConfig(
        session_continuation_mode=types.SessionContinuationMode.RESUME,
        conversation_id="12345678901234567890123456789012",
    )

  def test_session_continuation_validation_missing_id_raises(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaises(ValueError):
      ConcreteConfig(
          session_continuation_mode=types.SessionContinuationMode.RESUME,
          conversation_id=None,
      )

  def test_lightweight_method_returns_subclass_instance_with_presets(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(
        system_instructions="test prompt",
    ).lightweight()
    self.assertIsInstance(config, ConcreteConfig)
    self.assertEqual(config.system_instructions, "test prompt")
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertFalse(config.capabilities.enable_subagents)
    self.assertIsNotNone(config.compaction_config)
    self.assertEqual(config.compaction_config.token_threshold, 65536)

  def test_lightweight_method_preserves_explicit_compaction_config(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    user_compaction = types.CompactionConfig(
        token_threshold=12345,
    )
    config = ConcreteConfig(
        compaction_config=user_compaction,
    ).lightweight()
    self.assertEqual(config.compaction_config.token_threshold, 12345)

  def test_lightweight_method_merges_with_custom_capabilities(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(
        workspaces=["/tmp/workspace"],
        app_data_dir="/tmp/app",
        capabilities=types.CapabilitiesConfig(
            compaction_threshold=4000,
        ),
    ).lightweight()
    self.assertIsInstance(config, ConcreteConfig)
    self.assertEqual(config.workspaces, ["/tmp/workspace"])
    self.assertEqual(config.app_data_dir, "/tmp/app")
    self.assertEqual(config.capabilities.compaction_threshold, 4000)
    self.assertFalse(config.capabilities.enable_subagents)
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )

  def test_lightweight_method_returns_copy_without_mutating_original(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    original = ConcreteConfig()
    lightweight_config = original.lightweight()
    self.assertIsNot(lightweight_config, original)
    self.assertNotEqual(
        original.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        lightweight_config.capabilities.agent_behavior,
        types.AgentBehavior.MINIMAL,
    )

  def test_capabilities_none_raises_validation_error(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaises(pydantic.ValidationError):
      ConcreteConfig(capabilities=None)

  def test_lightweight_method_respects_custom_preset_overrides(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(
        capabilities=types.CapabilitiesConfig(
            agent_behavior=types.AgentBehavior.INTERACTIVE,
            enable_subagents=True,
        ),
    ).lightweight()
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.INTERACTIVE
    )
    self.assertTrue(config.capabilities.enable_subagents)
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )

  def test_lightweight_method_filters_disabled_tools_from_minimal_presets(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(
        capabilities=types.CapabilitiesConfig(
            disabled_tools=[types.BuiltinTools.RUN_COMMAND],
        ),
    ).lightweight()
    self.assertNotIn(
        types.BuiltinTools.RUN_COMMAND, config.capabilities.enabled_tools
    )
    self.assertIn(
        types.BuiltinTools.VIEW_FILE, config.capabilities.enabled_tools
    )
    self.assertIsNone(config.capabilities.disabled_tools)

  def test_compute_lightweight_presets_classmethod(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    presets = ConcreteConfig._compute_lightweight_presets()
    self.assertIn("capabilities", presets)
    self.assertEqual(
        presets["capabilities"].enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertEqual(
        presets["capabilities"].agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertFalse(presets["capabilities"].enable_subagents)
    self.assertIn("compaction_config", presets)
    self.assertEqual(presets["compaction_config"].token_threshold, 65536)

  def test_compute_lightweight_presets_with_dict_capabilities(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    presets = ConcreteConfig._compute_lightweight_presets({
        "capabilities": {"disabled_tools": [types.BuiltinTools.RUN_COMMAND]}
    })
    self.assertNotIn(
        types.BuiltinTools.RUN_COMMAND, presets["capabilities"].enabled_tools
    )
    self.assertIn(
        types.BuiltinTools.VIEW_FILE, presets["capabilities"].enabled_tools
    )

  def test_compute_lightweight_presets_with_dict_compaction_threshold(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    presets = ConcreteConfig._compute_lightweight_presets({
        "capabilities": {"compaction_threshold": 3000}
    })
    self.assertEqual(presets["capabilities"].compaction_threshold, 3000)
    self.assertNotIn("compaction_config", presets)

  def test_eval_method_returns_subclass_instance_with_presets(self):
    class ConcreteConfig(connection.AgentConfig):
      models: list[types.ModelTarget] = pydantic.Field(
          default_factory=lambda: [
              types.ModelTarget(
                  name="gemini-3-flash-preview",
                  endpoint=types.GeminiAPIEndpoint(),
              )
          ]
      )

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(
        system_instructions="test prompt",
    ).eval()
    self.assertIsInstance(config, ConcreteConfig)
    self.assertEqual(config.system_instructions, "test prompt")
    self.assertFalse(config.capabilities.enable_subagents)
    self.assertIsNotNone(config.capabilities.run_command_config)
    self.assertTrue(config.capabilities.run_command_config.enable_daemons)
    self.assertEqual(
        config.capabilities.disabled_tools,
        [types.BuiltinTools.GENERATE_IMAGE],
    )
    self.assertIsNone(config.capabilities.enabled_tools)
    self.assertEqual(config.policies, [policy.allow_all()])
    self.assertEqual(config.retry_config, types.RetryConfig.benchmark())
    self.assertEqual(
        config.models[0].endpoint.options.thinking_level,
        types.ThinkingLevel.HIGH,
    )

  def test_eval_method_preserves_explicit_overrides(self):
    class ConcreteConfig(connection.AgentConfig):
      models: list[types.ModelTarget] = pydantic.Field(
          default_factory=lambda: [
              types.ModelTarget(
                  name="gemini-3-flash-preview",
                  endpoint=types.GeminiAPIEndpoint(),
              )
          ]
      )

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    custom_policy = policy.Policy(
        name="custom", tool="run_command", decision=policy.Decision.DENY
    )
    custom_retry = types.RetryConfig(
        api_retry=types.ModelAPIRetryConfig(max_retries=3)
    )
    config = ConcreteConfig(
        policies=[custom_policy],
        retry_config=custom_retry,
        capabilities=types.CapabilitiesConfig(
            enable_subagents=True,
            disabled_tools=[types.BuiltinTools.SEARCH_WEB],
            run_command_config=types.RunCommandConfig(
                enable_daemons=False,
                timeout_seconds=120.0,
            ),
        ),
    ).eval()
    self.assertEqual(config.policies, [custom_policy])
    self.assertEqual(config.retry_config, custom_retry)
    self.assertTrue(config.capabilities.enable_subagents)
    self.assertIsNotNone(config.capabilities.run_command_config)
    self.assertFalse(config.capabilities.run_command_config.enable_daemons)
    self.assertEqual(
        config.capabilities.run_command_config.timeout_seconds, 120.0
    )
    self.assertEqual(
        config.capabilities.disabled_tools,
        [types.BuiltinTools.SEARCH_WEB],
    )

  def test_eval_method_with_enabled_tools_drops_disabled_tools(self):
    class ConcreteConfig(connection.AgentConfig):
      models: list[types.ModelTarget] = pydantic.Field(
          default_factory=lambda: [
              types.ModelTarget(
                  name="gemini-3-flash-preview",
                  endpoint=types.GeminiAPIEndpoint(),
              )
          ]
      )

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(
        capabilities=types.CapabilitiesConfig(
            enabled_tools=[types.BuiltinTools.VIEW_FILE],
        ),
    ).eval()
    self.assertEqual(
        config.capabilities.enabled_tools, [types.BuiltinTools.VIEW_FILE]
    )
    self.assertIsNone(config.capabilities.disabled_tools)
    self.assertFalse(config.capabilities.enable_subagents)

  def test_eval_raises_when_models_empty_or_missing_text_target(self):
    class NoModelsConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaisesRegex(
        ValueError,
        "Cannot apply thinking_level in eval\\(\\) on NoModelsConfig",
    ):
      NoModelsConfig().eval()

    # Passing thinking_level=None on a config without models succeeds.
    no_models_cfg = NoModelsConfig().eval(thinking_level=None)
    self.assertFalse(no_models_cfg.capabilities.enable_subagents)

    class ModelsConfig(connection.AgentConfig):
      models: list[types.ModelTarget] | None = None

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaisesRegex(
        ValueError, "only supported on configs with Gemini or Vertex"
    ):
      ModelsConfig(models=[]).eval()

    image_only = types.ModelTarget(
        name="imagen",
        types=[types.ModelType.IMAGE],
        endpoint=types.GeminiAPIEndpoint(),
    )
    with self.assertRaisesRegex(
        ValueError, "no ModelType.TEXT target found in models"
    ):
      ModelsConfig(models=[image_only]).eval()

    # Passing thinking_level=None skips thinking_level application and succeeds.
    cfg = ModelsConfig(models=[image_only]).eval(thinking_level=None)
    self.assertEqual(len(cfg.models), 1)
    self.assertIsNone(cfg.models[0].endpoint.options)

  def test_eval_raises_when_text_target_has_none_endpoint(self):
    class ModelsConfig(connection.AgentConfig):
      models: list[types.ModelTarget] | None = None

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaisesRegex(
        ValueError,
        "endpoint must be a GeminiAPIEndpoint or VertexEndpoint, got NoneType",
    ):
      ModelsConfig(
          models=[types.ModelTarget(name="text-no-endpoint", endpoint=None)]
      ).eval()

  def test_eval_raises_when_text_target_already_sets_thinking_level(self):
    class ModelsConfig(connection.AgentConfig):
      models: list[types.ModelTarget] | None = None

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    preconfigured = types.ModelTarget(
        name="preconfigured-model",
        endpoint=types.GeminiAPIEndpoint(
            options=types.GeminiModelOptions(
                thinking_level=types.ThinkingLevel.LOW
            )
        ),
    )
    with self.assertRaisesRegex(
        ValueError, "already sets thinking_level=.*pass it to .eval"
    ):
      ModelsConfig(models=[preconfigured]).eval()

    with self.assertRaisesRegex(
        ValueError, "already sets thinking_level=.*pass it to .eval"
    ):
      ModelsConfig(models=[preconfigured]).eval(
          thinking_level=types.ThinkingLevel.MEDIUM
      )

    # Passing thinking_level=None preserves the ModelTarget's thinking_level.
    preserved = ModelsConfig(models=[preconfigured]).eval(thinking_level=None)
    self.assertEqual(
        preserved.models[0].endpoint.options.thinking_level,
        types.ThinkingLevel.LOW,
    )


class ResolveActiveToolsTest(unittest.TestCase):
  """Tests for resolve_active_tools helper."""

  def test_none_config_returns_default(self):
    expected = set(types.BuiltinTools.default())
    self.assertEqual(connection.resolve_active_tools(None), expected)

  def test_enabled_tools_overrides_defaults(self):
    cfg = types.CapabilitiesConfig(
        enabled_tools=[
            types.BuiltinTools.VIEW_FILE,
            types.BuiltinTools.ASK_QUESTION,
        ]
    )
    expected = {types.BuiltinTools.VIEW_FILE, types.BuiltinTools.ASK_QUESTION}
    self.assertEqual(connection.resolve_active_tools(cfg), expected)

  def test_disabled_tools_subtracts_from_default(self):
    cfg = types.CapabilitiesConfig(
        disabled_tools=[types.BuiltinTools.RUN_COMMAND]
    )
    expected = set(types.BuiltinTools.default()) - {
        types.BuiltinTools.RUN_COMMAND
    }
    self.assertEqual(connection.resolve_active_tools(cfg), expected)
    self.assertNotIn(
        types.BuiltinTools.ASK_QUESTION,
        connection.resolve_active_tools(cfg),
    )

  def test_custom_defaults(self):
    custom_defaults = [
        types.BuiltinTools.VIEW_FILE,
        types.BuiltinTools.LIST_DIR,
    ]
    cfg = types.CapabilitiesConfig(
        disabled_tools=[types.BuiltinTools.LIST_DIR]
    )
    result = connection.resolve_active_tools(cfg, defaults=custom_defaults)
    self.assertEqual(result, {types.BuiltinTools.VIEW_FILE})

  def test_subagent_capabilities(self):
    subagent_cfg = types.SubagentCapabilities(
        disabled_tools=[types.BuiltinTools.RUN_COMMAND]
    )
    expected = set(types.BuiltinTools.default()) - {
        types.BuiltinTools.RUN_COMMAND
    }
    self.assertEqual(
        connection.resolve_active_tools(subagent_cfg), expected
    )

  def test_default_agent_config_no_deprecation_warning(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      cfg = ConcreteConfig()
      effective = cfg._get_effective_compaction_config()
      compaction_warnings = [
          item
          for item in w
          if issubclass(item.category, DeprecationWarning)
          and "compaction_threshold" in str(item.message)
      ]
      self.assertEqual(compaction_warnings, [])
      self.assertIsNone(effective)

  def test_get_effective_compaction_config_legacy_threshold_warns_once(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with warnings.catch_warnings():
      warnings.simplefilter("ignore", DeprecationWarning)
      cfg = ConcreteConfig(
          capabilities=types.CapabilitiesConfig(compaction_threshold=50000)
      )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      effective = cfg._get_effective_compaction_config()
      compaction_warnings = [
          item
          for item in w
          if issubclass(item.category, DeprecationWarning)
          and "CapabilitiesConfig.compaction_threshold is deprecated"
          in str(item.message)
      ]
      self.assertEqual(len(compaction_warnings), 1)
      self.assertIsNotNone(effective)
      self.assertEqual(effective.token_threshold, 50000)


class SharedConnectionUtilsTest(unittest.TestCase):
  """Tests for shared path and step ID helpers in connection.py."""

  def test_normalize_wire_path(self):
    self.assertEqual(
        connection.normalize_wire_path("file:///tmp/workspace/foo.py"),
        "/tmp/workspace/foo.py",
    )
    self.assertEqual(
        connection.normalize_wire_path("cns://el-d/home/user/data"),
        "/cns/el-d/home/user/data",
    )
    self.assertEqual(
        connection.normalize_wire_path("/already/plain/path"),
        "/already/plain/path",
    )
    self.assertEqual(
        connection.normalize_wire_path("file:///tmp/ws/../../etc/passwd"),
        "/etc/passwd",
    )
    self.assertEqual(
        connection.normalize_wire_path(
            "file:///tmp/ws/%2e%2e/%2e%2e/etc/passwd"
        ),
        "/etc/passwd",
    )
    self.assertEqual(
        connection.normalize_wire_path(
            "cns://el-d/home/user/../../other/secret"
        ),
        "/cns/el-d/other/secret",
    )
    self.assertEqual(
        connection.normalize_wire_path(
            "cns://el-d/home/user/%2e%2e/%2e%2e/other/secret"
        ),
        "/cns/el-d/other/secret",
    )
    self.assertEqual(
        connection.normalize_wire_path("cns://el-d/../../tmp/ws/foo.py"),
        "/cns/tmp/ws/foo.py",
    )
    self.assertEqual(
        connection.normalize_wire_path("/tmp/ws/../../etc/passwd"),
        "/etc/passwd",
    )
    self.assertEqual(
        connection.normalize_wire_path("//tmp/ws/../../etc/passwd"),
        "/etc/passwd",
    )
    self.assertEqual(
        connection.normalize_wire_path("file:////tmp/ws/../../etc/passwd"),
        "/etc/passwd",
    )
    self.assertEqual(
        connection.normalize_wire_path("file:///%2fetc/passwd"),
        "/etc/passwd",
    )
    with mock.patch.object(connection.os, "path", ntpath):
      self.assertEqual(
          connection.normalize_wire_path(r"C:\Users\alice\ws\..\..\secret.txt"),
          r"C:\Users\secret.txt",
      )
      self.assertEqual(
          connection.normalize_wire_path("cns://el-d/home/user/../other/data"),
          "/cns/el-d/home/other/data",
      )
    self.assertEqual(
        connection.normalize_wire_path("file:///tmp/./ws/./foo.py"),
        "/tmp/ws/foo.py",
    )
    self.assertEqual(
        connection.normalize_wire_path("https://example.com/a/../b"),
        "https://example.com/a/../b",
    )
    self.assertEqual(connection.normalize_wire_path(""), "")

  def test_normalize_workspace_paths(self):
    self.assertEqual(connection.normalize_workspace_paths(None), [])
    self.assertEqual(
        connection.normalize_workspace_paths(None, default_to_cwd=True),
        [os.getcwd()],
    )
    self.assertEqual(
        connection.normalize_workspace_paths(pathlib.Path("/tmp/ws")),
        [str(pathlib.Path("/tmp/ws").resolve())],
    )
    self.assertEqual(
        connection.normalize_workspace_paths(["cns://el-d/home/ws"]),
        ["/cns/el-d/home/ws"],
    )
    with self.assertRaises(ValueError):
      connection.normalize_workspace_paths(cast(Any, 123))

  def test_make_step_id(self):
    self.assertEqual(connection.make_step_id("traj-1", 4), "traj-1:4")
    self.assertEqual(connection.make_step_id("", 4), "4")


if __name__ == "__main__":
  unittest.main()
