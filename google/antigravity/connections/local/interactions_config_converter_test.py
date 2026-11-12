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

"""Unit tests for interactions_config_converter."""

import base64
import dataclasses
import functools
from typing import Any, cast

from absl.testing import absltest

from google.antigravity import types
from google.antigravity.connections.local import interactions_config_converter
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy
from google.antigravity.tools import tool_runner

_DeferredSection = types.SystemInstructionSection


def add_numbers(a: int, b: int) -> int:
  """Adds two numbers."""
  return a + b


def _deferred_tag_message(prefix: "str", body: "str") -> "str":
  """Tags a message with a fixed prefix."""
  return f"{prefix}:{body}"


class _DeferredFunctor:

  def __call__(self, value: "int", section: "_DeferredSection") -> "str":
    """Formats a section with an integer value."""
    return f"{value}:{section.content}"


class _ClassDocFunctor:
  """Class-level functor documentation."""

  def __call__(self, query: "str") -> "str":
    return query


@dataclasses.dataclass
class _DataclassNoDocFunctor:
  scale: int

  def __call__(self, count: "int") -> "int":
    return count * self.scale


class _UninspectableCallable:

  @property
  def __signature__(self):
    raise ValueError("no signature")

  def __call__(self, *args: Any, **kwargs: Any) -> None:
    del args, kwargs


class _NoOpPreTurnHook(hooks.PreTurnHook):

  async def run(self, context, data):
    del context, data
    return hooks.HookResult(allow=True)


class _NoOpPreToolHook(hooks.PreToolCallDecideHook):

  async def run(self, context, data):
    del context, data
    return hooks.HookResult(allow=True)


class _NoOpStopHook(hooks.StopHook):

  async def run(self, context, data):
    del context, data
    return hooks.StopResult(allow=True)


class InteractionsConfigConverterTest(absltest.TestCase):

  def test_create_interaction_session_continuation_modes(self):
    ev_create = interactions_config_converter.build_create_interaction_event(
        conversation_id="conv-123",
        session_continuation_mode=types.SessionContinuationMode.CREATE_ONLY,
    )
    self.assertEqual(ev_create["event_type"], "interaction.create")
    self.assertEqual(ev_create["agent"], "antigravity")
    self.assertEqual(ev_create["interaction_id"], "conv-123")
    self.assertNotIn("previous_interaction_id", ev_create)

    ev_resume = interactions_config_converter.build_create_interaction_event(
        conversation_id="conv-123",
        session_continuation_mode=types.SessionContinuationMode.RESUME,
    )
    self.assertNotIn("interaction_id", ev_resume)
    self.assertEqual(ev_resume["previous_interaction_id"], "conv-123")

    ev_both = interactions_config_converter.build_create_interaction_event(
        conversation_id="conv-123",
        session_continuation_mode=types.SessionContinuationMode.CREATE_OR_RESUME,
    )
    self.assertEqual(ev_both["interaction_id"], "conv-123")
    self.assertEqual(ev_both["previous_interaction_id"], "conv-123")

  def test_create_interaction_system_instructions(self):
    ev_custom = interactions_config_converter.build_create_interaction_event(
        system_instructions=types.CustomSystemInstructions(text="Be concise."),
    )
    self.assertEqual(
        ev_custom["developer_instructions"],
        [{"type": "text", "text": "Be concise."}],
    )

    ev_appended = interactions_config_converter.build_create_interaction_event(
        system_instructions=types.TemplatedSystemInstructions(
            identity="CustomBot",
            sections=[
                types.SystemInstructionSection(title="rules", content="Rule 1")
            ],
        ),
    )
    self.assertEqual(
        ev_appended["appended_developer_instructions"],
        {
            "custom_identity": "CustomBot",
            "appended_sections": [{"title": "rules", "content": "Rule 1"}],
        },
    )

    ev_str_sys = interactions_config_converter.build_create_interaction_event(
        system_instructions="Plain string instructions",
    )
    self.assertEqual(
        ev_str_sys["appended_developer_instructions"],
        {
            "appended_sections": [{
                "title": "user_system_instructions",
                "content": "Plain string instructions",
            }]
        },
    )

    with self.assertRaises(TypeError):
      interactions_config_converter._translate_system_instructions(
          cast(Any, 123),
          {},
      )

  def test_create_interaction_tools_and_mcp_servers(self):
    tr = tool_runner.ToolRunner([add_numbers])
    stdio_mcp = types.McpStdioServer(
        name="my_stdio",
        timeout_seconds=15,
        command="/bin/mcp",
        args=["--flag"],
        env={"FOO": "BAR"},
        enabled_tools=["tool_a"],
    )
    http_mcp = types.McpStreamableHttpServer(
        name="my_http",
        url="https://example.com/mcp",
        headers={"Auth": "Bearer x"},
        disabled_tools=["tool_b"],
    )
    caps = types.CapabilitiesConfig(
        enabled_tools=list(types.BuiltinTools),
        run_command_config=types.RunCommandConfig(
            timeout_seconds=5.0,
            enable_daemons=True,
            enable_sandbox=True,
        ),
    )

    ev = interactions_config_converter.build_create_interaction_event(
        tool_runner=tr,
        mcp_servers=[stdio_mcp, http_mcp],
        capabilities_config=caps,
    )
    tools = ev["tools"]
    fn_tools = [t for t in tools if t.get("type") == "function"]
    self.assertLen(fn_tools, 1)
    self.assertEqual(fn_tools[0]["name"], "add_numbers")
    self.assertEqual(fn_tools[0]["description"], "Adds two numbers.")
    self.assertIn("a", fn_tools[0]["parameters"]["properties"])
    self.assertIn("b", fn_tools[0]["parameters"]["properties"])

    self.assertIn(
        {
            "type": "mcp_server",
            "name": "my_stdio",
            "timeout": "15s",
            "stdio": {
                "command": "/bin/mcp",
                "args": ["--flag"],
                "env": {"FOO": "BAR"},
            },
            "allowed_tools": [{"mode": "any", "tools": ["tool_a"]}],
        },
        tools,
    )
    self.assertIn(
        {
            "type": "mcp_server",
            "name": "my_http",
            "http": {
                "url": "https://example.com/mcp",
                "headers": {"Auth": "Bearer x"},
            },
            "allowed_tools": [{"mode": "none", "tools": ["tool_b"]}],
        },
        tools,
    )
    self.assertIn(
        {
            "type": "bash",
            "max_timeout_ms": 5000,
            "enable_daemon_commands": True,
            "enable_sandbox": True,
        },
        tools,
    )
    self.assertIn(
        {
            "type": "filesystem",
            "supported_operations": [
                "file_read",
                "file_write",
                "file_edit",
                "file_find",
                "directory_list",
                "file_grep",
            ],
        },
        tools,
    )
    for expected_type in (
        "manage_task",
        "schedule",
        "google_search",
        "url_context",
    ):
      self.assertIn({"type": expected_type}, tools)
    self.assertEqual(
        ev["agent_config"]["policy"],
        {
            "enable_user_questions": True,
            "enable_image_generation": True,
        },
    )

  def test_root_vs_subagent_tool_partitioning_and_callable_variants(self):
    def sub_only_tool(q: str) -> str:
      """Subagent-only tool."""
      return q

    def schema_fn(y: str) -> str:
      """Tool with explicit schema."""
      return y

    class _CallableObj:
      """Callable object without __name__."""

      def __call__(self, x: int) -> int:
        return x + 1

    callable_obj = _CallableObj()
    schema_tool = tool_runner.ToolWithSchema(
        schema_fn,
        input_schema={
            "type": "object",
            "properties": {"y": {"type": "string"}},
        },
    )
    self.assertEqual(
        interactions_config_converter.callable_to_function_tool_dict(
            callable_obj
        )["name"],
        "_CallableObj",
    )
    schema_dict = interactions_config_converter.callable_to_function_tool_dict(
        schema_tool
    )
    self.assertEqual(schema_dict["name"], "schema_fn")
    self.assertEqual(schema_dict["description"], "Tool with explicit schema.")
    self.assertIn("y", schema_dict["parameters"]["properties"])

    tr = tool_runner.ToolRunner([add_numbers, sub_only_tool])
    sub = types.SubagentConfig(
        name="worker",
        description="Worker subagent",
        tools=["sub_only_tool", schema_tool],
        skills_config=types.SubagentInheritSkillsConfig(
            skill_names=["skill_a"],
            extra_skills_paths=["/tmp/extra_skills"],
        ),
    )
    ev = interactions_config_converter.build_create_interaction_event(
        tool_runner=tr,
        subagents=[sub],
    )
    root_fn_names = [
        t["name"] for t in ev["tools"] if t.get("type") == "function"
    ]
    self.assertEqual(root_fn_names, ["add_numbers"])

    sa_cfg = ev["agent_config"]["subagents_config"]["custom_subagents"][0]
    sa_fn_names = [
        t["name"] for t in sa_cfg["tools"] if t.get("type") == "function"
    ]
    self.assertEqual(sa_fn_names, ["sub_only_tool", "schema_fn"])
    self.assertEqual(
        sa_cfg["skills_config"],
        {
            "inherit_config": {
                "skill_names": ["skill_a"],
                "extra_skills_paths": ["/tmp/extra_skills"],
            }
        },
    )

    # Explicit `tools` list resolves both string tool names and callables.
    ev_explicit = interactions_config_converter.build_create_interaction_event(
        tool_runner=tr,
        tools=["add_numbers", "unregistered_tool", schema_tool],
    )
    explicit_fn_names = [
        t["name"] for t in ev_explicit["tools"] if t.get("type") == "function"
    ]
    self.assertEqual(
        explicit_fn_names,
        ["add_numbers", "unregistered_tool", "schema_fn"],
    )

  def test_callable_functor_and_partial_preserve_deferred_annotations_and_docs(
      self,
  ):
    """Verifies schema extraction for functors and partials with deferred hints.

    What: Converts callable class instances, functools.partial wrappers, and
      uninspectable callables whose annotations are PEP 563 strings.
    Why: Wrapping non-function callables without functools.wraps drops the
      underlying __call__ or target function globals/annotations, causing
      FunctionDeclaration schema parsing to raise KeyError on stringified hints
      and lose __call__ or partial docstrings/names.
    How: Asserts resolved parameter schemas, tool names, and descriptions across
      functor, dataclass functor, functools.partial, and signature-less cases.
    """
    functor = _DeferredFunctor()
    tr = tool_runner.ToolRunner([functor])
    for runner in (None, tr):
      fn_dict = interactions_config_converter.callable_to_function_tool_dict(
          functor, tool_runner=runner
      )
      self.assertEqual(fn_dict["name"], "_DeferredFunctor")
      self.assertEqual(
          fn_dict["description"], "Formats a section with an integer value."
      )
      params = fn_dict["parameters"]
      self.assertEqual(params["type"], "object")
      self.assertEqual(params["properties"]["value"]["type"], "integer")
      self.assertEqual(params["properties"]["section"]["type"], "object")
      self.assertIn("content", params["properties"]["section"]["properties"])
      self.assertEqual(params["required"], ["value", "section"])

    class_doc_dict = (
        interactions_config_converter.callable_to_function_tool_dict(
            _ClassDocFunctor()
        )
    )
    self.assertEqual(class_doc_dict["name"], "_ClassDocFunctor")
    self.assertEqual(
        class_doc_dict["description"], "Class-level functor documentation."
    )
    self.assertEqual(
        class_doc_dict["parameters"]["properties"]["query"]["type"], "string"
    )

    dataclass_dict = (
        interactions_config_converter.callable_to_function_tool_dict(
            _DataclassNoDocFunctor(scale=3)
        )
    )
    self.assertEqual(dataclass_dict["name"], "_DataclassNoDocFunctor")
    self.assertNotIn("description", dataclass_dict)
    self.assertEqual(
        dataclass_dict["parameters"]["properties"]["count"]["type"], "integer"
    )

    partial_fn = functools.partial(_deferred_tag_message, "INFO")
    partial_dict = interactions_config_converter.callable_to_function_tool_dict(
        partial_fn
    )
    self.assertEqual(partial_dict["name"], "_deferred_tag_message")
    self.assertEqual(
        partial_dict["description"], "Tags a message with a fixed prefix."
    )
    self.assertEqual(set(partial_dict["parameters"]["properties"]), {"body"})
    self.assertEqual(
        partial_dict["parameters"]["properties"]["body"]["type"], "string"
    )

    uninspectable_dict = (
        interactions_config_converter.callable_to_function_tool_dict(
            _UninspectableCallable()
        )
    )
    self.assertEqual(uninspectable_dict["name"], "_UninspectableCallable")
    self.assertEqual(
        uninspectable_dict["parameters"],
        {"type": "object", "properties": {}},
    )

  def test_subagent_skills_config_and_model_validation(self):
    ev_none = interactions_config_converter.build_create_interaction_event(
        subagents=[
            types.SubagentConfig(
                name="s_none",
                description="No skills",
                skills_config=types.SubagentNoneSkillsConfig(),
            ),
            types.SubagentConfig(
                name="s_override",
                description="Override skills",
                skills_config=types.SubagentOverrideSkillsConfig(
                    skills_paths=["/tmp/custom_skills"]
                ),
            ),
        ]
    )
    custom_sas = ev_none["agent_config"]["subagents_config"]["custom_subagents"]
    self.assertEqual(custom_sas[0]["skills_config"], {"none_config": {}})
    self.assertEqual(
        custom_sas[1]["skills_config"],
        {"override_config": {"skills_paths": ["/tmp/custom_skills"]}},
    )

    with self.assertRaises(ValueError):
      interactions_config_converter.build_create_interaction_event(
          subagents=[
              types.SubagentConfig(
                  name="s_inline",
                  description="Inline skills unsupported",
                  skills_config=types.SubagentOverrideSkillsConfig(
                      inline_skills=[
                          types.InlineSkill(
                              name="sk", description="d", content="i"
                          )
                      ]
                  ),
              )
          ]
      )

    ev_model = interactions_config_converter.build_create_interaction_event(
        subagents=[
            types.SubagentConfig(
                name="s_model",
                description="Pinned model subagent",
                model="gemini-2.5-flash",
            )
        ]
    )
    sa_model = ev_model["agent_config"]["subagents_config"]["custom_subagents"][
        0
    ]
    self.assertEqual(sa_model["model"], {"name": "gemini-2.5-flash"})

  def test_create_interaction_agent_config_fields(self):
    hr = hook_runner.HookRunner(
        pre_turn_hooks=[_NoOpPreTurnHook()],
        pre_tool_call_decide_hooks=[_NoOpPreToolHook()],
        stop_hooks=[_NoOpStopHook()],
    )
    caps = types.CapabilitiesConfig(
        enable_subagents=True,
        allowed_subagents=["helper"],
        max_subagent_depth=2,
        finish_tool_schema_json='{"type": "object"}',
        agent_behavior=types.AgentBehavior.INTERACTIVE,
        tool_output_truncation_config=types.ToolOutputTruncationConfig(
            max_tokens=4096
        ),
    )
    sub = types.SubagentConfig(
        name="helper",
        description="Helper subagent",
        capabilities=types.SubagentCapabilities(
            enabled_tools=[types.BuiltinTools.VIEW_FILE],
            agent_behavior=types.AgentBehavior.AUTONOMOUS,
        ),
    )
    model = types.ModelTarget(
        name="gemini-2.5-pro",
        types=[types.ModelType.TEXT],
        endpoint=types.GeminiAPIEndpoint(
            api_key="test-key",
            options=types.GeminiModelOptions(
                thinking_level=types.ThinkingLevel.EXTRA_HIGH,
                service_tier="priority",
            ),
        ),
    )
    vertex_model = types.ModelTarget(
        name="gemini-2.5-flash",
        types=[types.ModelType.IMAGE],
        endpoint=types.VertexEndpoint(
            base_url="https://us-central1-aiplatform.googleapis.com",
            http_headers={"X-Custom": "1"},
            api_key="vertex-key",
            project="my-proj",
            location="us-central1",
            options=types.GeminiModelOptions(
                thinking_level=types.ThinkingLevel.LOW,
            ),
        ),
    )

    ev = interactions_config_converter.build_create_interaction_event(
        app_data_dir="/tmp/appdata",
        skills_paths=["/tmp/skills"],
        workspaces=["/tmp/ws"],
        models=[model, vertex_model],
        hook_runner=hr,
        compaction_config=types.CompactionConfig(token_threshold=50000),
        capabilities_config=caps,
        subagents=[sub],
        retry_config=types.RetryConfig(
            api_retry=types.ModelAPIRetryConfig(
                max_retries=4,
                initial_sleep_duration_ms=250,
                exponential_multiplier=2.0,
            ),
            model_output_retry=types.ModelOutputRetryConfig(max_retries=2),
        ),
        budget_config=types.BudgetConfig(
            max_model_calls=10,
            scope=types.BudgetScope.FORWARD_LOOKING,
        ),
        policies=[
            policy.deny("run_command", reason="No bash"),
            policy.allow_all(),
        ],
        initial_trajectory=b"traj-bytes",
    )
    ac = ev["agent_config"]
    self.assertEqual(ac["type"], "antigravity")
    self.assertEqual(ac["app_data_dir"], "/tmp/appdata")
    self.assertEqual(ac["skills_paths"], ["/tmp/skills"])
    self.assertEqual(ac["enabled_hooks"], ["pre_turn", "pre_tool", "stop"])
    self.assertEqual(ac["compaction_config"], {"token_threshold": 50000})
    self.assertEqual(ac["finish_tool_output_schema"], {"type": "object"})
    self.assertEqual(ac["agent_behavior"], "interactive")
    self.assertEqual(
        ac["tool_output_truncation"],
        {"truncate": {"max_tokens": 4096}},
    )
    self.assertEqual(
        ac["retry_config"],
        {
            "api_retry": {
                "max_retries": 4,
                "initial_sleep_duration_ms": 250,
                "exponential_multiplier": 2.0,
            },
            "model_output_retry": {"max_retries": 2},
        },
    )
    self.assertEqual(
        ac["initial_trajectory"],
        base64.b64encode(b"traj-bytes").decode("ascii"),
    )
    self.assertEqual(
        ac["workspaces"], [{"filesystem_workspace": {"directory": "/tmp/ws"}}]
    )
    self.assertEqual(
        ac["models"],
        {
            "models": [
                {
                    "name": "gemini-2.5-pro",
                    "types": ["text"],
                    "gemini_api_endpoint": {
                        "api_key": "test-key",
                        "options": {
                            "thinking_level": "xhigh",
                            "service_tier": "priority",
                        },
                    },
                },
                {
                    "name": "gemini-2.5-flash",
                    "types": ["image"],
                    "vertex_endpoint": {
                        "base_url": (
                            "https://us-central1-aiplatform.googleapis.com"
                        ),
                        "http_headers": {"X-Custom": "1"},
                        "api_key": "vertex-key",
                        "project": "my-proj",
                        "location": "us-central1",
                        "options": {
                            "thinking_level": "low",
                        },
                    },
                },
            ]
        },
    )
    self.assertEqual(
        ac["subagents_config"]["allowed_subagents"],
        {"enumerated": {"names": ["helper"]}},
    )
    self.assertEqual(ac["subagents_config"]["max_nesting_depth"], 2)
    self.assertEqual(
        ac["subagents_config"]["custom_subagents"],
        [{
            "name": "helper",
            "description": "Helper subagent",
            "tools": [{
                "type": "filesystem",
                "supported_operations": ["file_read"],
            }],
            "allowed_subagents": {"disabled": {}},
            "agent_behavior": "autonomous",
        }],
    )
    self.assertEqual(
        ac["budget_config"],
        {"max_model_calls": 10, "scope": "forward_looking"},
    )
    self.assertEqual(
        ac["policy_config"],
        {
            "workspace_containment": "disabled",
            "rules": [
                {
                    "tool": "run_command",
                    "name": "run_command",
                    "decision": "deny",
                    "deny_reason": "No bash",
                },
                {
                    "tool": "*",
                    "name": "allow_all",
                    "decision": "allow",
                },
            ],
        },
    )

  def test_create_interaction_policy_workspace_only_and_mcp_rule(self):
    ev = interactions_config_converter.build_create_interaction_event(
        policies=[
            policy.workspace_only(workspaces=["/tmp/ws"]),
            policy.allow("my_mcp/tool_a"),
            policy.allow_all(),
        ],
    )
    pc = ev["agent_config"]["policy_config"]
    self.assertNotIn("workspace_containment", pc)
    self.assertEqual(
        pc["rules"],
        [
            {
                "tool": "view_file",
                "name": "workspace_only",
                "decision": "deny",
            },
            {
                "tool": "create_file",
                "name": "workspace_only",
                "decision": "deny",
            },
            {
                "tool": "edit_file",
                "name": "workspace_only",
                "decision": "deny",
            },
            {
                "server_name": "my_mcp",
                "tool": "tool_a",
                "name": "my_mcp/tool_a",
                "decision": "allow",
            },
            {
                "tool": "*",
                "name": "allow_all",
                "decision": "allow",
            },
        ],
    )

  def test_create_interaction_rejects_dynamic_and_auto_policies(self):
    with self.assertRaises(types.AntigravityValidationError):
      interactions_config_converter.build_create_interaction_event(
          policies=[policy.auto()],
      )

    with self.assertRaises(types.AntigravityValidationError):
      interactions_config_converter.build_create_interaction_event(
          policies=[
              policy.deny(
                  "run_command",
                  when=lambda args: "rm" in args.get("CommandLine", ""),
              )
          ],
      )

  def test_content_to_user_input_event(self):
    ev_str = interactions_config_converter.content_to_user_input_event(
        "Hello\x00world"
    )
    self.assertEqual(
        ev_str,
        {
            "event_type": "input",
            "content": [{"type": "text", "text": "Helloworld"}],
        },
    )

    ev_sc = interactions_config_converter.content_to_user_input_event(
        types.SlashCommand(name=types.BuiltinSlashCommandName.PLAN)
    )
    self.assertEqual(
        ev_sc,
        {
            "event_type": "input",
            "input_content": [{"type": "slash_command", "name": "plan"}],
        },
    )

    ev_sc_text = interactions_config_converter.content_to_user_input_event([
        "Please",
        types.SlashCommand(name=types.BuiltinSlashCommandName.PLAN),
        "write a script",
    ])
    self.assertEqual(
        ev_sc_text,
        {
            "event_type": "input",
            "input_content": [
                {"type": "text", "text": "Please"},
                {"type": "slash_command", "name": "plan"},
                {"type": "text", "text": "write a script"},
            ],
        },
    )

    img = types.Image(data=b"pngdata", mime_type="image/png")
    ev_multi = interactions_config_converter.content_to_user_input_event(
        ["Describe this:", img]
    )
    self.assertEqual(
        ev_multi,
        {
            "event_type": "input",
            "content": [
                {"type": "text", "text": "Describe this:"},
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": base64.b64encode(b"pngdata").decode("ascii"),
                },
            ],
        },
    )

  def test_tool_result_to_function_result_event(self):
    ev_ok = interactions_config_converter.tool_result_to_function_result_event(
        call_id="call-1",
        tool_name="my_tool",
        result_dict={"sum": 42},
    )
    self.assertEqual(
        ev_ok,
        {
            "event_type": "function_result",
            "call_id": "call-1",
            "name": "my_tool",
            "is_error": False,
            "result": {"sum": 42},
        },
    )

    ev_err = interactions_config_converter.tool_result_to_function_result_event(
        call_id="call-2",
        tool_name="my_tool",
        error_message="boom",
    )
    self.assertEqual(
        ev_err,
        {
            "event_type": "function_result",
            "call_id": "call-2",
            "name": "my_tool",
            "is_error": True,
            "result": "boom",
        },
    )

  def test_question_responses_to_elicitation_result_events(self):
    events = interactions_config_converter.question_responses_to_elicitation_result_events(
        elicitation_ids=["q-0", "q-1"],
        responses=[
            types.QuestionResponse(
                selected_option_ids=["1"], freeform_response="extra"
            ),
            types.QuestionResponse(skipped=True),
        ],
    )
    self.assertEqual(
        events,
        [
            {
                "event_type": "elicitation_result",
                "elicitation_id": "q-0",
                "multiple_choice": {
                    "selected_choice_labels": ["1"],
                    "user_input": [{"type": "text", "text": "extra"}],
                },
            },
            {
                "event_type": "elicitation_result",
                "elicitation_id": "q-1",
                "multiple_choice": {},
            },
        ],
    )

    skipped_events = interactions_config_converter.question_responses_to_elicitation_result_events(
        elicitation_ids=["q-0"],
        responses=[],
        cancelled=False,
    )
    self.assertEqual(
        skipped_events,
        [{
            "event_type": "elicitation_result",
            "elicitation_id": "q-0",
            "multiple_choice": {},
        }],
    )

    cancelled_events = interactions_config_converter.question_responses_to_elicitation_result_events(
        elicitation_ids=["q-0", "q-1"],
        cancelled=True,
    )
    self.assertEqual(
        cancelled_events,
        [{
            "event_type": "elicitation_result",
            "elicitation_id": "q-0",
            "confirmation": {"is_confirmed": False},
        }],
    )

  def test_cancel_and_complete_events(self):
    self.assertEqual(
        interactions_config_converter.build_cancel_interaction_event(),
        {"event_type": "interaction.cancel"},
    )
    self.assertEqual(
        interactions_config_converter.build_complete_interaction_event(),
        {"event_type": "interaction.complete"},
    )


if __name__ == "__main__":
  absltest.main()
