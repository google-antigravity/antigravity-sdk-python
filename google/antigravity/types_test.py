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

"""Tests for Google Antigravity SDK Pydantic type definitions.

Validates model construction, validation, immutability, forward compatibility,
and the AntigravityValidationError wrapper.
"""

import asyncio
from collections.abc import AsyncIterator
import pathlib
import tempfile
from typing import Any, cast
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pydantic

from google.antigravity import types
from google.antigravity.conversation import conversation


class ToolCallTest(unittest.TestCase):
  """Validates the ToolCall Pydantic model."""

  def test_basic_construction(self):
    """Verifies that a ToolCall can be constructed with name and args.

    What: Checks basic field assignment.
    Why: Validates the happy path for the most commonly used SDK type.
    How: Constructs a ToolCall and asserts field values.
    """
    tc = types.ToolCall(name="read_file", args={"path": "/tmp/foo"})
    self.assertEqual(tc.name, "read_file")
    self.assertEqual(tc.args, {"path": "/tmp/foo"})

  def test_canonical_path_default_none(self):
    """Verifies that canonical_path defaults to None."""
    tc = types.ToolCall(name="read_file", args={"path": "/tmp/foo"})
    self.assertIsNone(tc.canonical_path)

  def test_canonical_path_explicit(self):
    """Verifies that canonical_path can be explicitly set."""
    tc = types.ToolCall(
        name="read_file",
        args={"path": "/tmp/foo"},
        canonical_path="/tmp/foo",
    )
    self.assertEqual(tc.canonical_path, "/tmp/foo")

  def test_default_args(self):
    """Verifies that args defaults to empty dict when omitted.

    What: Checks default factory for args field.
    Why: Many tool calls have no arguments.
    How: Constructs a ToolCall without args and asserts empty dict.
    """
    tc = types.ToolCall(name="no_args_tool")
    self.assertEqual(tc.args, {})

  def test_id_defaults_to_none(self):
    """Verifies that id defaults to None when omitted."""
    tc = types.ToolCall(name="tool")
    self.assertIsNone(tc.id)

  def test_id_explicitly_set(self):
    """Verifies that id can be explicitly set."""
    tc = types.ToolCall(id="call_123", name="tool")
    self.assertEqual(tc.id, "call_123")

  def test_step_id_defaults_to_none(self):
    """Verifies that step_id defaults to None when omitted."""
    tc = types.ToolCall(name="tool")
    self.assertIsNone(tc.step_id)

  def test_step_id_explicitly_set(self):
    """Verifies that step_id can be explicitly set."""
    tc = types.ToolCall(step_id="traj-1:5", name="tool")
    self.assertEqual(tc.step_id, "traj-1:5")

  def test_extra_fields_ignored(self):
    """Verifies that unknown fields are silently dropped.

    What: Checks extra='ignore' behavior.
    Why: Forward compatibility — newer backends may add fields.
    How: Constructs a ToolCall with an unknown field and asserts it's absent.
    """
    tc = types.ToolCall(name="tool", **cast(Any, {"unknown_field": "value"}))
    self.assertFalse(hasattr(tc, "unknown_field"))

  def test_missing_name_raises(self):
    """Verifies that omitting required field 'name' raises.

    What: Checks required field validation.
    Why: Every tool call must have a name.
    How: Attempts construction without name and asserts ValidationError.
    """
    with self.assertRaises(pydantic.ValidationError):
      types.ToolCall()


class ToolResultTest(unittest.TestCase):
  """Validates the ToolResult Pydantic model."""

  def test_success_result(self):
    """Verifies construction of a successful ToolResult.

    What: Checks that result and error fields are set correctly.
    Why: Validates the common success case.
    How: Constructs a ToolResult with a result and asserts fields.
    """
    tr = types.ToolResult(name="sum_tool", result=42)
    self.assertEqual(tr.name, "sum_tool")
    self.assertEqual(tr.result, 42)
    self.assertIsNone(tr.error)

  def test_error_result(self):
    """Verifies construction of an error ToolResult.

    What: Checks that error field is populated.
    Why: Validates the error path for failed tool executions.
    How: Constructs a ToolResult with an error string and asserts.
    """
    tr = types.ToolResult(name="bad_tool", error="kaboom")
    self.assertEqual(tr.error, "kaboom")
    self.assertIsNone(tr.result)

  def test_mutable(self):
    """Verifies that ToolResult is mutable (not frozen).

    What: Checks that fields can be updated after construction.
    Why: ToolResult is built up during execution and may need mutation.
    How: Sets a field after construction and asserts the new value.
    """
    tr = types.ToolResult(name="tool")
    tr.result = "updated"
    self.assertEqual(tr.result, "updated")

  def test_id_defaults_to_none(self):
    """Verifies that id defaults to None when omitted."""
    tr = types.ToolResult(name="tool")
    self.assertIsNone(tr.id)

  def test_id_explicitly_set(self):
    """Verifies that id can be explicitly set for call correlation."""
    tr = types.ToolResult(id="call_123", name="tool", result="ok")
    self.assertEqual(tr.id, "call_123")

  def test_id_mutable(self):
    """Verifies that id can be set after construction."""
    tr = types.ToolResult(name="tool")
    tr.id = "call_456"
    self.assertEqual(tr.id, "call_456")

  def test_step_id_defaults_to_none(self):
    """Verifies that step_id defaults to None when omitted."""
    tr = types.ToolResult(name="tool")
    self.assertIsNone(tr.step_id)

  def test_step_id_explicitly_set(self):
    """Verifies that step_id can be explicitly set."""
    tr = types.ToolResult(step_id="traj-1:3", name="tool", result="ok")
    self.assertEqual(tr.step_id, "traj-1:3")

  def test_extra_fields_ignored(self):
    """Verifies extra='ignore' on ToolResult.

    What: Checks forward compatibility.
    Why: Consistent extra field handling across all models.
    How: Passes an unknown field and asserts it's not present.
    """
    tr = types.ToolResult(name="tool", **cast(Any, {"unknown": "value"}))
    self.assertFalse(hasattr(tr, "unknown"))


class StepTest(unittest.TestCase):
  """Validates the Step Pydantic model."""

  def test_basic_construction(self):
    """Verifies that a Step can be constructed with all fields."""
    tc = types.ToolCall(name="run_command", args={"cmd": "ls"})
    step = types.Step(
        id="1",
        step_index=1,
        type=types.StepType.TOOL_CALL,
        status=types.StepStatus.DONE,
        source=types.StepSource.MODEL,
        content="output",
        thinking="reasoning",
        tool_calls=[tc],
        error="",
    )
    self.assertEqual(step.id, "1")
    self.assertEqual(step.step_index, 1)
    self.assertEqual(step.type, types.StepType.TOOL_CALL)
    self.assertEqual(step.tool_calls[0].name, "run_command")

  def test_defaults(self):
    """Verifies that all Step fields have sensible defaults."""
    step = types.Step()
    self.assertEqual(step.id, "")
    self.assertEqual(step.step_index, 0)
    self.assertEqual(step.trajectory_id, "")
    self.assertEqual(step.parent_trajectory_id, "")
    self.assertEqual(step.depth, 0)
    self.assertEqual(step.type, types.StepType.UNKNOWN)
    self.assertEqual(step.status, types.StepStatus.UNKNOWN)
    self.assertEqual(step.source, types.StepSource.UNKNOWN)
    self.assertEqual(step.content, "")
    self.assertEqual(step.thinking, "")
    self.assertEqual(step.tool_calls, [])
    self.assertEqual(step.error, "")
    self.assertIsNone(step.is_complete_response)
    self.assertIsNone(step.structured_output)

  def test_mutable(self):
    """Verifies that Step is mutable as per Karmel's model."""
    step = types.Step(id="1", content="hello")
    step.content = "goodbye"
    self.assertEqual(step.content, "goodbye")

  def test_extra_fields_allowed(self):
    """Verifies extra='allow' on Step as per Karmel's model."""
    step = types.Step(id="1", future_field="value")
    self.assertTrue(hasattr(step, "future_field"))
    self.assertEqual(getattr(step, "future_field"), "value")

  def test_nested_tool_call(self):
    """Verifies that a Step can contain a nested ToolCall."""
    step = types.Step(
        id="5",
        type=types.StepType.TOOL_CALL,
        tool_calls=[{"name": "my_tool", "args": {"x": 1}}],
    )
    self.assertEqual(len(step.tool_calls), 1)
    self.assertEqual(step.tool_calls[0].name, "my_tool")
    self.assertEqual(step.tool_calls[0].args, {"x": 1})


class HookResultTest(unittest.TestCase):
  """Validates the HookResult Pydantic model."""

  def test_defaults(self):
    """Verifies that HookResult defaults to allow=True.

    What: Checks default values.
    Why: The default behavior should be permissive.
    How: Constructs a HookResult with no arguments and checks allow.
    """
    hr = types.HookResult()
    self.assertTrue(hr.allow)
    self.assertEqual(hr.message, "")

  def test_deny(self):
    """Verifies construction of a deny HookResult.

    What: Checks explicit deny behavior.
    Why: Validates the policy enforcement path.
    How: Constructs with allow=False and a message.
    """
    hr = types.HookResult(allow=False, message="blocked by policy")
    self.assertFalse(hr.allow)
    self.assertEqual(hr.message, "blocked by policy")

  def test_mutable(self):
    """Verifies that HookResult is mutable.

    What: Checks that allow can be changed after construction.
    Why: Hook runners may need to update results during dispatch.
    How: Modifies the allow field after construction.
    """
    hr = types.HookResult(allow=True)
    hr.allow = False
    self.assertFalse(hr.allow)

  def test_modified_args(self):
    """Verifies construction and default of modified_args on HookResult."""
    hr_default = types.HookResult()
    self.assertIsNone(hr_default.modified_args)

    hr = types.HookResult(allow=True, modified_args={"cmd": "echo safe"})
    self.assertTrue(hr.allow)
    self.assertEqual(hr.modified_args, {"cmd": "echo safe"})


class QuestionResponseTest(unittest.TestCase):
  """Validates the QuestionResponse Pydantic model."""

  def test_defaults(self):
    """Verifies QuestionResponse defaults.

    What: Checks that all fields have sensible defaults.
    Why: Most responses only populate one field.
    How: Constructs with no arguments and checks defaults.
    """
    qr = types.QuestionResponse()
    self.assertIsNone(qr.selected_option_ids)
    self.assertEqual(qr.freeform_response, "")
    self.assertFalse(qr.skipped)

  def test_skipped(self):
    """Verifies construction of a skipped response.

    What: Checks skipped flag.
    Why: Users can skip questions.
    How: Constructs with skipped=True and asserts.
    """
    qr = types.QuestionResponse(skipped=True)
    self.assertTrue(qr.skipped)

  def test_selected_options(self):
    """Verifies construction with selected option IDs.

    What: Checks option selection.
    Why: Most common response type is selecting from options.
    How: Constructs with selected_option_ids and asserts.
    """
    qr = types.QuestionResponse(selected_option_ids=["opt1", "opt2"])
    self.assertEqual(qr.selected_option_ids, ["opt1", "opt2"])

  def test_write_in(self):
    """Verifies construction with a write-in response.

    What: Checks write-in text.
    Why: Freeform text is an alternative to option selection.
    How: Constructs with freeform_response and asserts.
    """
    qr = types.QuestionResponse(freeform_response="custom answer")
    self.assertEqual(qr.freeform_response, "custom answer")


class QuestionHookResultTest(unittest.TestCase):
  """Validates the QuestionHookResult Pydantic model."""

  def test_basic_construction(self):
    """Verifies construction with a list of responses.

    What: Checks required field 'responses'.
    Why: Every interaction must have at least one response.
    How: Constructs with a list of QuestionResponse objects.
    """
    qhr = types.QuestionHookResult(
        responses=[types.QuestionResponse(skipped=True)]
    )
    self.assertEqual(len(qhr.responses), 1)
    self.assertTrue(qhr.responses[0].skipped)
    self.assertFalse(qhr.cancelled)

  def test_cancelled(self):
    """Verifies cancelled interaction.

    What: Checks cancelled flag.
    Why: User may cancel an interaction (e.g. EOF).
    How: Constructs with cancelled=True.
    """
    qhr = types.QuestionHookResult(responses=[], cancelled=True)
    self.assertTrue(qhr.cancelled)

  def test_missing_responses_raises(self):
    """Verifies that omitting required 'responses' raises.

    What: Checks required field validation.
    Why: responses is a required field.
    How: Attempts construction without responses.
    """
    with self.assertRaises(pydantic.ValidationError):
      types.QuestionHookResult()


class AntigravityValidationErrorTest(unittest.TestCase):
  """Validates the AntigravityValidationError wrapper."""

  def test_basic_construction(self):
    """Verifies direct construction with a message.

    What: Checks that the exception stores message and errors.
    Why: SDK consumers catch this instead of pydantic.ValidationError.
    How: Constructs the exception and checks attributes.
    """
    err = types.AntigravityValidationError("bad input")
    self.assertEqual(str(err), "bad input")
    self.assertEqual(err.message, "bad input")
    self.assertEqual(err.errors, [])

  def test_from_pydantic(self):
    """Verifies construction from a real Pydantic ValidationError.

    What: Checks the _from_pydantic factory method.
    Why: This is the primary construction path at SDK boundaries.
    How: Triggers a ValidationError and wraps it.
    """
    err = None
    try:
      types.ToolCall()  # Missing required 'name' field.
    except pydantic.ValidationError as e:
      err = e

    self.assertIsNotNone(err, "Expected ValidationError was not raised.")
    wrapped = types.AntigravityValidationError._from_pydantic(err)
    self.assertIn("name", wrapped.message)
    self.assertGreater(len(wrapped.errors), 0)

  def test_is_exception(self):
    """Verifies that AntigravityValidationError is a proper Exception.

    What: Checks isinstance relationship.
    Why: Must be catchable as a standard Python exception.
    How: Asserts isinstance against Exception.
    """
    err = types.AntigravityValidationError("test")
    self.assertIsInstance(err, Exception)

  def test_with_errors_list(self):
    """Verifies construction with an explicit errors list.

    What: Checks that the errors list is preserved.
    Why: Structured errors allow programmatic handling.
    How: Passes an explicit errors list and asserts.
    """
    errors = [{"type": "missing", "loc": ("name",), "msg": "Field required"}]
    err = types.AntigravityValidationError("validation failed", errors=errors)
    self.assertEqual(len(err.errors), 1)
    self.assertEqual(err.errors[0]["type"], "missing")

  def test_step_pydantic(self):
    """Tests that Step can be instantiated as a Pydantic model."""
    step = types.Step(id="1", content="test content")
    self.assertEqual(step.id, "1")
    self.assertEqual(step.content, "test content")
    self.assertEqual(step.type, types.StepType.UNKNOWN)
    self.assertEqual(step.tool_calls, [])

  def test_step_with_tool_calls(self):
    """Tests that Step can hold multiple tool calls."""
    tc1 = types.ToolCall(name="tool1")
    tc2 = types.ToolCall(name="tool2")
    step = types.Step(tool_calls=[tc1, tc2])
    self.assertEqual(len(step.tool_calls), 2)
    self.assertEqual(step.tool_calls[0].name, "tool1")
    self.assertEqual(step.tool_calls[1].name, "tool2")

  def test_tool_call_pydantic(self):
    """Tests that ToolCall can be instantiated as a Pydantic model."""
    tc = types.ToolCall(name="my_tool", args={"a": 1})
    self.assertEqual(tc.name, "my_tool")
    self.assertEqual(tc.args, {"a": 1})


class AskQuestionModelsTest(unittest.TestCase):
  """Tests for AskQuestion related models."""

  def test_ask_question_option(self):
    opt = types.AskQuestionOption(id="A", text="Option A")
    self.assertEqual(opt.id, "A")
    self.assertEqual(opt.text, "Option A")

  def test_ask_question_entry(self):
    opt = types.AskQuestionOption(id="A", text="Option A")
    entry = types.AskQuestionEntry(question="Q?", options=[opt])
    self.assertEqual(entry.question, "Q?")
    self.assertEqual(len(entry.options), 1)
    self.assertFalse(entry.is_multi_select)

  def test_ask_question_interaction_spec(self):
    opt = types.AskQuestionOption(id="A", text="Option A")
    entry = types.AskQuestionEntry(question="Q?", options=[opt])
    spec = types.AskQuestionInteractionSpec(questions=[entry])
    self.assertEqual(len(spec.questions), 1)


class ThinkingLevelTest(unittest.TestCase):
  """Tests for the ThinkingLevel enum."""

  def test_enum_values(self):
    """Verifies each enum member has the expected string value."""
    self.assertEqual(types.ThinkingLevel.MINIMAL, "minimal")
    self.assertEqual(types.ThinkingLevel.LOW, "low")
    self.assertEqual(types.ThinkingLevel.MEDIUM, "medium")
    self.assertEqual(types.ThinkingLevel.HIGH, "high")

  def test_string_comparison(self):
    """Verifies ThinkingLevel members compare equal to their string values."""
    self.assertEqual(types.ThinkingLevel.LOW, "low")
    self.assertNotEqual(types.ThinkingLevel.LOW, "high")


class ModelTargetTest(unittest.TestCase):
  """Tests for the polymorphic ModelTarget hierarchy."""

  def test_basic_text_construction(self):
    options = types.GeminiModelOptions(
        thinking_level=types.ThinkingLevel.HIGH,
    )
    endpoint = types.GeminiAPIEndpoint(api_key="test-key", options=options)
    mc = types.ModelTarget(
        name="gemini-pro",
        types=[types.ModelType.TEXT],
        endpoint=endpoint,
    )
    self.assertEqual(mc.name, "gemini-pro")
    self.assertEqual(mc.types, [types.ModelType.TEXT])
    self.assertIsInstance(mc.endpoint, types.GeminiAPIEndpoint)
    self.assertEqual(mc.endpoint.api_key, "test-key")
    self.assertEqual(
        mc.endpoint.options.thinking_level, types.ThinkingLevel.HIGH
    )

  def test_vertex_endpoint_construction(self):
    endpoint = types.VertexEndpoint(project="my-proj", location="us-east1")
    mc = types.ModelTarget(
        name="gemini-ultra",
        types=[types.ModelType.TEXT],
        endpoint=endpoint,
    )
    self.assertIsInstance(mc.endpoint, types.VertexEndpoint)
    self.assertEqual(mc.endpoint.project, "my-proj")
    self.assertEqual(mc.endpoint.location, "us-east1")

  def test_image_model_construction(self):
    mc = types.ModelTarget(
        name="imagen-3",
        types=[types.ModelType.IMAGE],
    )
    self.assertEqual(mc.types, [types.ModelType.IMAGE])


class SystemInstructionsTest(unittest.TestCase):
  """Tests for the SystemInstructions Pydantic model union."""

  def test_custom_construction(self):
    """Verifies construction of CustomSystemInstructions."""
    si = types.CustomSystemInstructions(text="Override all defaults.")
    self.assertEqual(si.text, "Override all defaults.")

  def test_templated_construction(self):
    """Verifies construction of TemplatedSystemInstructions."""
    section = types.SystemInstructionSection(
        title="extra", content="More instructions"
    )
    si = types.TemplatedSystemInstructions(
        identity="New Identity", sections=[section]
    )
    self.assertEqual(si.identity, "New Identity")
    self.assertEqual(len(si.sections), 1)
    self.assertEqual(si.sections[0].title, "extra")

  def test_union_parsing_custom(self):
    """Verifies that Pydantic parses CustomSystemInstructions from dict."""
    data = {"text": "Be helpful."}
    adapter = pydantic.TypeAdapter(types.SystemInstructions)
    si = adapter.validate_python(data)
    self.assertIsInstance(si, types.CustomSystemInstructions)
    self.assertEqual(si.text, "Be helpful.")

  def test_union_parsing_templated(self):
    """Verifies that Pydantic parses TemplatedSystemInstructions from dict."""
    data = {
        "identity": "I am robot",
        "sections": [{"title": "rules", "content": "Do no harm"}],
    }
    adapter = pydantic.TypeAdapter(types.SystemInstructions)
    si = adapter.validate_python(data)
    self.assertIsInstance(si, types.TemplatedSystemInstructions)
    self.assertEqual(si.identity, "I am robot")
    self.assertEqual(len(si.sections), 1)
    self.assertEqual(si.sections[0].title, "rules")

  def test_custom_text_is_required(self):
    """Verifies that CustomSystemInstructions raises when text is missing."""
    with self.assertRaises(pydantic.ValidationError):
      types.CustomSystemInstructions()  # type: ignore

  def test_templated_empty_construction(self):
    """Verifies that TemplatedSystemInstructions can be constructed empty."""
    si = types.TemplatedSystemInstructions()
    self.assertIsNone(si.identity)
    self.assertEqual(si.sections, [])

  def test_union_parsing_empty_dict(self):
    """Verifies that Pydantic parses empty dict as TemplatedSystemInstructions."""
    data = {}
    adapter = pydantic.TypeAdapter(types.SystemInstructions)
    si = adapter.validate_python(data)
    self.assertIsInstance(si, types.TemplatedSystemInstructions)
    self.assertIsNone(si.identity)
    self.assertEqual(si.sections, [])


class BuiltinToolsTest(parameterized.TestCase):
  """Tests for the BuiltinTools enum."""

  @parameterized.named_parameters(
      ("list_dir", types.BuiltinTools.LIST_DIR, "list_directory"),
      ("search_dir", types.BuiltinTools.SEARCH_DIR, "search_directory"),
      ("find_file", types.BuiltinTools.FIND_FILE, "find_file"),
      ("view_file", types.BuiltinTools.VIEW_FILE, "view_file"),
      ("create_file", types.BuiltinTools.CREATE_FILE, "create_file"),
      ("edit_file", types.BuiltinTools.EDIT_FILE, "edit_file"),
      ("run_command", types.BuiltinTools.RUN_COMMAND, "run_command"),
      ("ask_question", types.BuiltinTools.ASK_QUESTION, "ask_question"),
      ("search_web", types.BuiltinTools.SEARCH_WEB, "search_web"),
      (
          "read_url_content",
          types.BuiltinTools.READ_URL_CONTENT,
          "read_url_content",
      ),
      ("start_subagent", types.BuiltinTools.START_SUBAGENT, "start_subagent"),
      ("generate_image", types.BuiltinTools.GENERATE_IMAGE, "generate_image"),
      ("finish", types.BuiltinTools.FINISH, "finish"),
  )
  def test_enum_values(self, enum_member, expected_value):
    """Verifies each enum member has the expected string value."""
    self.assertEqual(enum_member, expected_value)

  def test_read_only_covers_all_tools(self):
    """Verifies read_only + write tools = full enum.

    If a new BuiltinTools member is added without updating either read_only()
    or this test's write_tools set, the test will fail, forcing the developer
    to categorize the new tool.
    """
    read_only = set(types.BuiltinTools.read_only())
    write_tools = {
        types.BuiltinTools.CREATE_FILE,
        types.BuiltinTools.EDIT_FILE,
        types.BuiltinTools.RUN_COMMAND,
        types.BuiltinTools.ASK_QUESTION,
        types.BuiltinTools.START_SUBAGENT,
        types.BuiltinTools.GENERATE_IMAGE,
        types.BuiltinTools.SEARCH_WEB,
    }
    self.assertEqual(
        read_only | write_tools,
        set(types.BuiltinTools),
        "A new BuiltinTools member was added but not categorized in"
        " read_only() or this test's write_tools set.",
    )
    self.assertFalse(
        read_only & write_tools,
        "read_only and write_tools must not overlap.",
    )

  def test_nondestructive_covers_all_tools(self):
    """Verifies nondestructive + destructive tools = full enum.

    If a new BuiltinTools member is added without updating either
    nondestructive() or this test's destructive_tools set, the test will fail,
    forcing the developer to categorize the new tool.
    """
    nondestructive = set(types.BuiltinTools.nondestructive())
    destructive_tools = {
        types.BuiltinTools.RUN_COMMAND,
    }
    self.assertEqual(
        nondestructive | destructive_tools,
        set(types.BuiltinTools),
        "A new BuiltinTools member was added but not categorized in"
        " nondestructive() or this test's destructive_tools set.",
    )
    self.assertFalse(
        nondestructive & destructive_tools,
        "nondestructive and destructive_tools must not overlap.",
    )

  def test_all_tools_returns_every_member(self):
    """Verifies that all_tools() returns every enum member."""
    self.assertEqual(
        set(types.BuiltinTools.all_tools()), set(types.BuiltinTools)
    )
    self.assertEqual(
        len(types.BuiltinTools.all_tools()), len(types.BuiltinTools)
    )

  def test_none_returns_empty_list(self):
    """Verifies that none() returns an empty list."""
    self.assertEqual(types.BuiltinTools.none(), [])

  def test_minimal_returns_six_minimal_tools(self):
    """Verifies that minimal() returns exactly the 6 core software engineering tools."""
    expected = [
        types.BuiltinTools.RUN_COMMAND,
        types.BuiltinTools.VIEW_FILE,
        types.BuiltinTools.CREATE_FILE,
        types.BuiltinTools.EDIT_FILE,
        types.BuiltinTools.LIST_DIR,
        types.BuiltinTools.SEARCH_DIR,
    ]
    self.assertEqual(types.BuiltinTools.minimal(), expected)


class AgentBehaviorTest(unittest.TestCase):
  """Tests for the AgentBehavior enum."""

  def test_enum_values(self):
    self.assertEqual(types.AgentBehavior.AUTONOMOUS, "autonomous")
    self.assertEqual(types.AgentBehavior.INTERACTIVE, "interactive")
    self.assertEqual(types.AgentBehavior.MINIMAL, "minimal")


class CapabilitiesConfigTest(unittest.TestCase):
  """Tests for the CapabilitiesConfig Pydantic model."""

  def test_default_construction(self):
    """Verifies defaults: subagents enabled, no tool lists, no threshold."""
    config = types.CapabilitiesConfig()
    self.assertTrue(config.enable_subagents)
    self.assertEqual(config.agent_behavior, types.AgentBehavior.AUTONOMOUS)
    self.assertIsNone(config.enabled_tools)
    self.assertIsNone(config.disabled_tools)
    self.assertIsNone(config.compaction_threshold)
    self.assertIsNone(config.finish_tool_schema_json)

  def test_agent_behavior_explicit(self):
    """Verifies that agent_behavior can be explicitly set via enum or string."""
    config = types.CapabilitiesConfig(
        agent_behavior=types.AgentBehavior.INTERACTIVE
    )
    self.assertEqual(config.agent_behavior, types.AgentBehavior.INTERACTIVE)
    config_str = types.CapabilitiesConfig(agent_behavior="interactive")
    self.assertEqual(config_str.agent_behavior, types.AgentBehavior.INTERACTIVE)
    config_min = types.CapabilitiesConfig(
        agent_behavior=types.AgentBehavior.MINIMAL
    )
    self.assertEqual(config_min.agent_behavior, types.AgentBehavior.MINIMAL)
    config_min_str = types.CapabilitiesConfig(agent_behavior="minimal")
    self.assertEqual(config_min_str.agent_behavior, types.AgentBehavior.MINIMAL)

  def test_enabled_tools(self):
    """Verifies that enabled_tools accepts a list of BuiltinTools."""
    config = types.CapabilitiesConfig(
        enabled_tools=[types.BuiltinTools.VIEW_FILE]
    )
    self.assertEqual(config.enabled_tools, [types.BuiltinTools.VIEW_FILE])
    self.assertIsNone(config.disabled_tools)

  def test_disabled_tools(self):
    """Verifies that disabled_tools accepts a list of BuiltinTools."""
    config = types.CapabilitiesConfig(
        disabled_tools=[
            types.BuiltinTools.RUN_COMMAND,
        ]
    )
    self.assertIsNone(config.enabled_tools)
    self.assertEqual(len(config.disabled_tools), 1)

  def test_mutually_exclusive_raises(self):
    """Verifies that setting both enabled_tools and disabled_tools raises."""
    with self.assertRaises(pydantic.ValidationError):
      types.CapabilitiesConfig(
          enabled_tools=[types.BuiltinTools.VIEW_FILE],
          disabled_tools=[types.BuiltinTools.RUN_COMMAND],
      )

  def test_compaction_threshold_explicit(self):
    """Verifies that compaction_threshold accepts an explicit integer."""
    config = types.CapabilitiesConfig(compaction_threshold=50000)
    self.assertEqual(config.compaction_threshold, 50000)

  def test_ask_question_warning_when_not_interactive(self):
    """Verifies warning when ASK_QUESTION is enabled and not interactive."""
    with self.assertLogs(level="WARNING") as log_cm:
      types.CapabilitiesConfig(
          enabled_tools=[types.BuiltinTools.ASK_QUESTION],
          agent_behavior=types.AgentBehavior.AUTONOMOUS,
      )
    self.assertTrue(
        any("ASK_QUESTION is enabled" in msg for msg in log_cm.output)
    )

  def test_ask_question_no_warning_when_interactive(self):
    """Verifies no warning when ASK_QUESTION is enabled in interactive mode."""
    with mock.patch("logging.warning") as mock_warn:
      types.CapabilitiesConfig(
          enabled_tools=[types.BuiltinTools.ASK_QUESTION],
          agent_behavior=types.AgentBehavior.INTERACTIVE,
      )
      mock_warn.assert_not_called()

  def test_max_subagent_depth_and_allowed_subagents(self):
    """Verifies max_subagent_depth and allowed_subagents on CapabilitiesConfig."""
    config = types.CapabilitiesConfig(
        max_subagent_depth=3,
        allowed_subagents=["researcher", "reviewer"],
    )
    self.assertEqual(config.max_subagent_depth, 3)
    self.assertEqual(config.allowed_subagents, ["researcher", "reviewer"])

  def test_max_subagent_depth_ge_1_enforced(self):
    """Verifies max_subagent_depth < 1 raises ValidationError."""
    with self.assertRaises(pydantic.ValidationError) as cm:
      types.CapabilitiesConfig(max_subagent_depth=0)
    self.assertIn("greater than or equal to 1", str(cm.exception))

  def test_run_command_config_defaults_and_custom(self):
    """Verifies RunCommandConfig defaults and custom settings on CapabilitiesConfig."""
    config_default = types.CapabilitiesConfig()
    self.assertIsNone(config_default.run_command_config)

    run_cmd_default = types.RunCommandConfig()
    self.assertFalse(run_cmd_default.enable_daemons)
    self.assertIsNone(run_cmd_default.timeout_seconds)
    self.assertFalse(run_cmd_default.enable_sandbox)

    config_custom = types.CapabilitiesConfig(
        run_command_config=types.RunCommandConfig(
            enable_daemons=True,
            timeout_seconds=600.0,
            enable_sandbox=True,
        )
    )
    self.assertIsNotNone(config_custom.run_command_config)
    self.assertTrue(config_custom.run_command_config.enable_daemons)
    self.assertEqual(config_custom.run_command_config.timeout_seconds, 600.0)
    self.assertTrue(config_custom.run_command_config.enable_sandbox)

    subagent_caps = types.SubagentCapabilities(
        run_command_config=types.RunCommandConfig(timeout_seconds=30.0)
    )
    self.assertIsNotNone(subagent_caps.run_command_config)
    self.assertEqual(subagent_caps.run_command_config.timeout_seconds, 30.0)

  def test_run_command_config_non_positive_timeout_raises(self):
    """Verifies timeout_seconds <= 0 raises ValidationError."""
    with self.assertRaises(pydantic.ValidationError):
      types.RunCommandConfig(timeout_seconds=0)

    with self.assertRaises(pydantic.ValidationError):
      types.RunCommandConfig(timeout_seconds=-10.0)

  def test_subagent_capabilities_allowed_subagents(self):
    """Verifies allowed_subagents on SubagentCapabilities."""
    caps = types.SubagentCapabilities(allowed_subagents=["fact_checker"])
    self.assertEqual(caps.allowed_subagents, ["fact_checker"])

  def test_max_subagent_depth_fails_when_subagents_disabled(self):
    """Verifies ValidationError when max_subagent_depth is set but subagents disabled."""
    with self.assertRaises(pydantic.ValidationError) as cm:
      types.CapabilitiesConfig(
          enable_subagents=False,
          max_subagent_depth=2,
      )
    self.assertIn("max_subagent_depth cannot be configured", str(cm.exception))

  def test_max_subagent_depth_1_fails_when_subagents_disabled(self):
    """Verifies max_subagent_depth=1 also fails when subagents disabled."""
    with self.assertRaises(pydantic.ValidationError) as cm:
      types.CapabilitiesConfig(
          enable_subagents=False,
          max_subagent_depth=1,
      )
    self.assertIn("max_subagent_depth cannot be configured", str(cm.exception))

  def test_allowed_subagents_fails_when_subagents_disabled(self):
    """Verifies ValidationError when allowed_subagents is set but subagents disabled."""
    with self.assertRaises(pydantic.ValidationError) as cm:
      types.CapabilitiesConfig(
          enable_subagents=False,
          allowed_subagents=["worker"],
      )
    self.assertIn("allowed_subagents cannot be specified", str(cm.exception))

  def test_allowed_subagents_empty_list_fails_when_subagents_disabled(self):
    """Verifies ValidationError when allowed_subagents=[] and subagents disabled."""
    with self.assertRaises(pydantic.ValidationError) as cm:
      types.CapabilitiesConfig(
          enable_subagents=False,
          allowed_subagents=[],
      )
    self.assertIn("allowed_subagents cannot be specified", str(cm.exception))

  def test_max_subagent_depth_fails_when_start_subagent_tool_disabled(self):
    """Verifies ValidationError when START_SUBAGENT is in disabled_tools."""
    with self.assertRaises(pydantic.ValidationError) as cm:
      types.CapabilitiesConfig(
          disabled_tools=[types.BuiltinTools.START_SUBAGENT],
          max_subagent_depth=3,
      )
    self.assertIn("max_subagent_depth cannot be configured", str(cm.exception))

  def test_allowed_subagents_fails_when_start_subagent_tool_disabled(self):
    """Verifies ValidationError when START_SUBAGENT is in disabled_tools and allowed_subagents is set."""
    with self.assertRaises(pydantic.ValidationError) as cm:
      types.CapabilitiesConfig(
          disabled_tools=[types.BuiltinTools.START_SUBAGENT],
          allowed_subagents=["worker"],
      )
    self.assertIn("allowed_subagents cannot be specified", str(cm.exception))

  def test_max_subagent_depth_fails_when_start_subagent_not_in_enabled_tools(
      self,
  ):
    """Verifies ValidationError when START_SUBAGENT is omitted from enabled_tools."""
    with self.assertRaises(pydantic.ValidationError) as cm:
      types.CapabilitiesConfig(
          enabled_tools=[types.BuiltinTools.VIEW_FILE],
          max_subagent_depth=2,
      )
    self.assertIn("max_subagent_depth cannot be configured", str(cm.exception))

  def test_allowed_subagents_fails_when_start_subagent_not_in_enabled_tools(
      self,
  ):
    """Verifies ValidationError when START_SUBAGENT is omitted from enabled_tools and allowed_subagents is set."""
    with self.assertRaises(pydantic.ValidationError) as cm:
      types.CapabilitiesConfig(
          enabled_tools=[types.BuiltinTools.VIEW_FILE],
          allowed_subagents=["worker"],
      )
    self.assertIn("allowed_subagents cannot be specified", str(cm.exception))

  def test_subagent_ask_question_warning_when_not_interactive(self):
    """Verifies that a warning is logged for SubagentCapabilities."""
    with self.assertLogs(level="WARNING") as log_cm:
      types.SubagentCapabilities(
          enabled_tools=[types.BuiltinTools.ASK_QUESTION],
          agent_behavior=types.AgentBehavior.AUTONOMOUS,
      )
    self.assertTrue(
        any(
            "ASK_QUESTION is enabled on subagent" in msg
            for msg in log_cm.output
        )
    )


class AntigravityConnectionErrorTest(unittest.TestCase):
  """Validates the AntigravityConnectionError hierarchy."""

  def test_is_exception(self):
    """Verifies that AntigravityConnectionError is a proper Exception."""
    err = types.AntigravityConnectionError("connection failed")
    self.assertIsInstance(err, Exception)

  def test_message(self):
    """Verifies that the message is stored and retrievable."""
    err = types.AntigravityConnectionError("timeout")
    self.assertEqual(str(err), "timeout")


class ToolExecutionErrorTest(unittest.TestCase):
  """Validates the ToolExecutionError exception class."""

  def test_basic_construction(self):
    """Verifies construction with required arguments and defaults."""
    err = types.ToolExecutionError("command failed", tool_name="run_command")
    self.assertIsInstance(err, RuntimeError)
    self.assertEqual(str(err), "command failed")
    self.assertEqual(err.tool_name, "run_command")
    self.assertIsNone(err.server_name)
    self.assertIsNone(err.call_id)
    self.assertIsNone(err.step_id)

  def test_explicit_server_name_and_call_id(self):
    """Verifies construction with explicit server_name and call_id."""
    err = types.ToolExecutionError(
        "query failed",
        tool_name="mcp_tool",
        server_name="mcp_server",
        call_id="call_123",
        step_id="step_456",
    )
    self.assertIsInstance(err, RuntimeError)
    self.assertEqual(str(err), "query failed")
    self.assertEqual(err.tool_name, "mcp_tool")
    self.assertEqual(err.server_name, "mcp_server")
    self.assertEqual(err.call_id, "call_123")
    self.assertEqual(err.step_id, "step_456")


class ImageTest(unittest.TestCase):
  """Tests for the Image content attachment primitive and its validators."""

  def test_basic_construction(self):
    """Verifies that an Image can be successfully constructed with valid arguments."""
    img = types.Image(
        data=b"png_data", mime_type="image/png", description="diagram"
    )
    self.assertEqual(img.data, b"png_data")
    self.assertEqual(img.mime_type, "image/png")
    self.assertEqual(img.description, "diagram")

  def test_unsupported_mime_type_raises(self):
    """Verifies that an unsupported Image MIME type triggers ValidationError."""
    with self.assertRaises(pydantic.ValidationError):
      types.Image(data=b"gif_bytes", mime_type="image/gif")

  def test_from_file_success(self):
    """Verifies that from_file loader loads bytes and guesses MIME correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "photo.png"
      fake_bytes = b"png_file_content"
      tmp_file.write_bytes(fake_bytes)

      img = types.Image.from_file(tmp_file, description="profile photo")
      self.assertEqual(img.data, fake_bytes)
      self.assertEqual(img.mime_type, "image/png")
      self.assertEqual(img.description, "profile photo")

  def test_from_file_inference_failure_raises(self):
    """Verifies that from_file loader raises ValueError when extension is unknown."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "photo.unknown"
      fake_bytes = b"some_bytes"
      tmp_file.write_bytes(fake_bytes)

      with self.assertRaisesRegex(
          ValueError, "Could not infer a valid MIME type"
      ):
        types.Image.from_file(tmp_file)


class DocumentTest(parameterized.TestCase):
  """Tests for the Document content attachment primitive and its validators."""

  def test_basic_construction(self):
    """Verifies that a Document can be successfully constructed with valid arguments."""
    doc = types.Document(
        data=b"pdf_data", mime_type="application/pdf", description="report"
    )
    self.assertEqual(doc.data, b"pdf_data")
    self.assertEqual(doc.mime_type, "application/pdf")
    self.assertEqual(doc.description, "report")

  @parameterized.parameters(
      "application/pdf",
      "application/json",
      "text/css",
      "text/csv",
      "text/html",
      "text/javascript",
      "text/plain",
      "text/rtf",
      "text/xml",
  )
  def test_supported_mime_types(self, mime_type: str):
    """Verifies that all supported Document MIME types pass validation."""
    doc = types.Document(data=b"sample_data", mime_type=mime_type)
    self.assertEqual(doc.mime_type, mime_type)

  def test_unsupported_mime_type_raises(self):
    """Verifies that an unsupported Document MIME type triggers ValidationError."""
    with self.assertRaises(pydantic.ValidationError):
      types.Document(data=b"image_bytes", mime_type="image/png")

  def test_from_file_success(self):
    """Verifies that from_file loader loads bytes and guesses MIME correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "report.pdf"
      fake_bytes = b"pdf_file_content"
      tmp_file.write_bytes(fake_bytes)

      doc = types.Document.from_file(tmp_file, description="monthly report")
      self.assertEqual(doc.data, fake_bytes)
      self.assertEqual(doc.mime_type, "application/pdf")
      self.assertEqual(doc.description, "monthly report")


class AudioTest(parameterized.TestCase):
  """Validates the Audio content attachment primitive and its validators."""

  def test_basic_construction(self):
    """Verifies that an Audio can be successfully constructed with valid arguments."""
    audio = types.Audio(data=b"mp3_data", mime_type="audio/mp3")
    self.assertEqual(audio.data, b"mp3_data")
    self.assertEqual(audio.mime_type, "audio/mp3")

  @parameterized.parameters(
      "audio/wav",
      "audio/x-wav",
      "audio/wave",
      "audio/vnd.wave",
      "audio/mp3",
      "audio/mp4",
      "audio/webm",
      "audio/aac",
      "audio/ogg",
      "audio/flac",
      "audio/opus",
      "audio/mpeg",
      "audio/m4a",
      "audio/l16",
  )
  def test_supported_mime_types(self, mime_type: str):
    """Verifies that all supported Audio MIME types pass validation."""
    audio = types.Audio(data=b"sample_data", mime_type=mime_type)
    self.assertEqual(audio.mime_type, mime_type)

  def test_unsupported_mime_type_raises(self):
    """Verifies that an unsupported Audio MIME type triggers ValidationError."""
    with self.assertRaises(pydantic.ValidationError):
      types.Audio(data=b"wav_bytes", mime_type="audio/unsupported-wav")


class VideoTest(unittest.TestCase):
  """Validates the Video content attachment primitive and its validators."""

  def test_basic_construction(self):
    """Verifies that a Video can be successfully constructed with valid arguments."""
    video = types.Video(data=b"mp4_data", mime_type="video/mp4")
    self.assertEqual(video.data, b"mp4_data")
    self.assertEqual(video.mime_type, "video/mp4")

  def test_unsupported_mime_type_raises(self):
    """Verifies that an unsupported Video MIME type triggers ValidationError."""
    with self.assertRaises(pydantic.ValidationError):
      types.Video(data=b"mov_bytes", mime_type="video/unsupported-mov")


class ContentFromFileResolverTest(parameterized.TestCase):
  """Validates the global from_file content resolver helper function."""

  @parameterized.named_parameters(
      ("image", "diagram.png", types.Image, "image/png"),
      ("document", "report.pdf", types.Document, "application/pdf"),
      ("audio", "clip.mp3", types.Audio, "audio/mpeg"),
      ("video", "movie.mp4", types.Video, "video/mp4"),
  )
  def test_resolves_from_file(self, filename, expected_class, expected_mime):
    """Verifies that local files are resolved to the correct Content primitives."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / filename
      tmp_file.write_bytes(b"fake_bytes")

      res = types.from_file(tmp_file, description="test asset")
      self.assertIsInstance(res, expected_class)
      self.assertEqual(res.mime_type, expected_mime)
      self.assertEqual(res.description, "test asset")

  def test_non_existent_path_raises_error(self):
    """Verifies that passing a non-existent path triggers FileNotFoundError."""
    with self.assertRaises(FileNotFoundError):
      types.from_file("non_existent_file_xyz.png")

  def test_directory_path_raises_error(self):
    """Verifies that passing a path targeting a directory triggers IsADirectoryError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaises(IsADirectoryError):
        types.from_file(tmpdir)

  def test_inference_failure_raises_clear_value_error(self):
    """Verifies that an extensionless path failure raises a descriptive ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "extensionless_data_blob"
      tmp_file.write_bytes(b"some_anonymous_bytes")

      with self.assertRaisesRegex(
          ValueError, "Could not infer a valid MIME type"
      ):
        types.from_file(tmp_file)

  def test_permission_error_wrapping(self):
    """Verifies that permission errors are caught and wrapped appropriately."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "locked.png"
      tmp_file.write_bytes(b"data")

      # Mock read_bytes to raise PermissionError
      with mock.patch.object(
          pathlib.Path,
          "read_bytes",
          autospec=True,
          side_effect=PermissionError("access denied"),
      ):
        with self.assertRaisesRegex(PermissionError, "Permission denied"):
          types.from_file(tmp_file)

  def test_os_error_wrapping(self):
    """Verifies that OS-level read errors are caught and wrapped appropriately."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "faulty.png"
      tmp_file.write_bytes(b"data")

      # Mock read_bytes to raise OSError
      with mock.patch.object(
          pathlib.Path,
          "read_bytes",
          autospec=True,
          side_effect=OSError("disk read failure"),
      ):
        with self.assertRaisesRegex(OSError, "Failed to read file"):
          types.from_file(tmp_file)


class ContentFromBytesResolverTest(parameterized.TestCase):
  """Validates the global from_bytes content resolver helper function."""

  @parameterized.named_parameters(
      ("image", "image/png", types.Image),
      ("document", "application/pdf", types.Document),
      ("audio", "audio/mpeg", types.Audio),
      ("video", "video/mp4", types.Video),
  )
  def test_resolves_from_bytes(self, mime_type, expected_class):
    """Verifies that bytes and MIME type are resolved to the correct Content primitives."""
    res = types.from_bytes(
        b"fake_bytes", mime_type=mime_type, description="byte asset"
    )
    self.assertIsInstance(res, expected_class)
    self.assertEqual(res.mime_type, mime_type)
    self.assertEqual(res.description, "byte asset")
    self.assertEqual(res.data, b"fake_bytes")

  def test_unsupported_mime_type_raises_error(self):
    """Verifies that an unsupported MIME type raises a descriptive ValueError."""
    with self.assertRaisesRegex(ValueError, "Unsupported MIME type"):
      types.from_bytes(b"data", mime_type="application/x-unsupported-custom")


class ChatResponseStreamTest(unittest.IsolatedAsyncioTestCase):
  """Tests for ChatResponse async stream and caching properties."""

  async def test_text_concatenation(self):
    """Verifies that text() aggregates and concatenates all Text chunks."""
    t1 = types.Text(step_index=1, text="Hello ")
    t2 = types.Thought(step_index=1, text="internal reasoning...")
    t3 = types.Text(step_index=2, text="world!")

    async def mock_stream():
      yield t1
      yield t2
      yield t3

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    self.assertEqual(await response.text(), "Hello world!")

  async def test_thoughts_sugared_stream(self):
    """Verifies thoughts property yields only Thought delta strings."""
    t1 = types.Thought(step_index=1, text="Let me think...")
    t2 = types.Text(step_index=2, text="Standard response text.")

    async def mock_stream():
      yield t1
      yield t2

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    thoughts = [token async for token in response.thoughts]
    self.assertEqual(thoughts, ["Let me think..."])

  async def test_tool_calls_sugared_stream(self):
    """Verifies tool_calls property yields only ToolCall objects."""
    t1 = types.Text(step_index=1, text="Invoking tool...")
    t2 = types.ToolCall(id="call_1", name="get_weather", args={})

    async def mock_stream():
      yield t1
      yield t2

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    calls = [call async for call in response.tool_calls]
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0].name, "get_weather")

  async def test_lazy_caching_and_re_iteration(self):
    """Verifies that stream tokens are cached in memory and replayed for re-iteration."""
    t1 = types.Text(step_index=1, text="A")
    t2 = types.Text(step_index=2, text="B")

    pull_count = 0

    async def mock_stream():
      nonlocal pull_count
      pull_count += 1
      yield t1
      yield t2

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )

    # Iteration Round 1 (pulls from live stream and caches)
    with self.subTest("round_1_live_caching"):
      round_1 = [token async for token in response]
      self.assertEqual(round_1, ["A", "B"])
      self.assertEqual(pull_count, 1)

    # Iteration Round 2 (replays from buffered cache without pulling again)
    with self.subTest("round_2_cache_replay"):
      round_2 = [token async for token in response]
      self.assertEqual(round_2, ["A", "B"])
      self.assertEqual(pull_count, 1)

  async def test_sequential_thoughts_then_text(self):
    """Cache replay: thoughts first, then text sees all Text chunks."""
    chunks = [
        types.Thought(step_index=1, text="hmm..."),
        types.Text(step_index=2, text="Hello "),
        types.Text(step_index=3, text="world!"),
    ]

    async def mock_stream():
      for c in chunks:
        yield c

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    thoughts = [t async for t in response.thoughts]
    self.assertEqual(thoughts, ["hmm..."])
    text_deltas = [t async for t in response]
    self.assertEqual(text_deltas, ["Hello ", "world!"])

  async def test_sequential_text_then_thoughts(self):
    """Cache replay: text first, then thoughts sees all Thought chunks."""
    chunks = [
        types.Thought(step_index=1, text="thinking"),
        types.Text(step_index=2, text="answer"),
    ]

    async def mock_stream():
      for c in chunks:
        yield c

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    text_deltas = [t async for t in response]
    self.assertEqual(text_deltas, ["answer"])
    thoughts = [t async for t in response.thoughts]
    self.assertEqual(thoughts, ["thinking"])

  async def test_sequential_tool_calls_then_text(self):
    """Cache replay: tool_calls first, then text."""
    tc = types.ToolCall(id="c1", name="search", args={"q": "x"})
    chunks = [
        types.Text(step_index=1, text="Let me search."),
        tc,
        types.Text(step_index=3, text="Done."),
    ]

    async def mock_stream():
      for c in chunks:
        yield c

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    calls = [c async for c in response.tool_calls]
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0].name, "search")
    text_deltas = [t async for t in response]
    self.assertEqual(text_deltas, ["Let me search.", "Done."])

  async def test_all_three_sequential(self):
    """Cache replay: thoughts → tool_calls → text from one response."""
    chunks = [
        types.Thought(step_index=1, text="plan"),
        types.Text(step_index=2, text="I'll search."),
        types.ToolCall(id="c1", name="search", args={}),
        types.Text(step_index=4, text="Found it."),
    ]

    async def mock_stream():
      for c in chunks:
        yield c

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    thoughts = [t async for t in response.thoughts]
    calls = [c async for c in response.tool_calls]
    text_deltas = [t async for t in response]
    self.assertEqual(thoughts, ["plan"])
    self.assertEqual(len(calls), 1)
    self.assertEqual(text_deltas, ["I'll search.", "Found it."])

  async def test_two_chunks_cursors_independent(self):
    """Two raw .chunks cursors both see identical output."""
    chunks = [
        types.Thought(step_index=1, text="a"),
        types.Text(step_index=2, text="b"),
    ]

    async def mock_stream():
      for c in chunks:
        yield c

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    cursor1 = [c async for c in response.chunks]
    cursor2 = [c async for c in response.chunks]
    self.assertEqual(cursor1, cursor2)
    self.assertEqual(len(cursor1), 2)

  async def test_resolve_then_text_then_aiter(self):
    """resolve(), text(), and __aiter__ all work on the same response."""
    chunks = [
        types.Thought(step_index=1, text="think"),
        types.Text(step_index=2, text="hello "),
        types.Text(step_index=3, text="world"),
    ]

    async def mock_stream():
      for c in chunks:
        yield c

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    resolved = await response.resolve()
    self.assertEqual(len(resolved), 3)
    full_text = await response.text()
    self.assertEqual(full_text, "hello world")
    text_deltas = [t async for t in response]
    self.assertEqual(text_deltas, ["hello ", "world"])

  async def test_interleaved_chunk_types(self):
    """Thought-Text-Thought-Text-ToolCall pattern streams correctly."""
    chunks = [
        types.Thought(step_index=1, text="A"),
        types.Text(step_index=2, text="B"),
        types.Thought(step_index=3, text="C"),
        types.Text(step_index=4, text="D"),
        types.ToolCall(id="c1", name="fn", args={}),
    ]

    async def mock_stream():
      for c in chunks:
        yield c

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    all_chunks = [c async for c in response.chunks]
    self.assertEqual(len(all_chunks), 5)
    thoughts = [t async for t in response.thoughts]
    self.assertEqual(thoughts, ["A", "C"])
    text_deltas = [t async for t in response]
    self.assertEqual(text_deltas, ["B", "D"])
    calls = [c async for c in response.tool_calls]
    self.assertEqual(len(calls), 1)

  async def test_error_propagation_to_all_cursors(self):
    """Error storage: stream error re-raised to every cursor."""

    async def mock_stream():
      yield types.Text(step_index=1, text="ok")
      raise RuntimeError("network failure")

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )

    # First cursor: sees one chunk then the error.
    with self.assertRaises(RuntimeError):
      _ = [c async for c in response.chunks]

    # Second cursor: replays cached chunk, then re-raises stored error.
    with self.assertRaises(RuntimeError):
      _ = [c async for c in response.chunks]

    # Sugared iterator: same behavior.
    with self.assertRaises(RuntimeError):
      _ = [t async for t in response]

  async def test_concurrent_cursors_via_gather(self):
    """Lock safety: two cursors via asyncio.gather don't crash."""
    chunks = [
        types.Thought(step_index=1, text="t"),
        types.Text(step_index=2, text="a"),
        types.Text(step_index=3, text="b"),
    ]

    async def mock_stream():
      for c in chunks:
        yield c

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )

    async def drain_thoughts():
      return [t async for t in response.thoughts]

    async def drain_text():
      return [t async for t in response]

    thoughts, text_deltas = await asyncio.gather(drain_thoughts(), drain_text())
    self.assertEqual(thoughts, ["t"])
    self.assertEqual(text_deltas, ["a", "b"])

  async def test_explicit_lock_contention_double_check(self):
    """Explicitly tests the double-check condition under lock contention."""

    class InstrumentedLock(asyncio.Lock):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.second_call = asyncio.Event()

      async def acquire(self) -> bool:
        self.calls += 1
        if self.calls == 2:
          self.second_call.set()
        return await super().acquire()

    event_a_pulling = asyncio.Event()
    event_b_waiting = asyncio.Event()

    async def slow_stream():
      event_a_pulling.set()
      await event_b_waiting.wait()
      yield types.Text(step_index=1, text="chunk1")

    response = types.ChatResponse(
        slow_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )

    instrumented_lock = InstrumentedLock()
    response._pull_lock = instrumented_lock

    cursor_a = response.chunks
    cursor_b = response.chunks

    task_a = asyncio.create_task(cursor_a.__anext__())
    await event_a_pulling.wait()

    task_b = asyncio.create_task(cursor_b.__anext__())
    await instrumented_lock.second_call.wait()

    event_b_waiting.set()

    chunk_a = await task_a
    chunk_b = await task_b

    self.assertEqual(chunk_a, chunk_b)
    self.assertEqual(chunk_a.text, "chunk1")

  async def test_empty_stream_all_iterators(self):
    """All iterators return empty on an empty stream."""

    async def mock_empty_stream():
      return
      yield  # Makes this an async generator.

    response = types.ChatResponse(
        mock_empty_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    self.assertEqual([c async for c in response.chunks], [])
    self.assertEqual([t async for t in response.thoughts], [])
    self.assertEqual([t async for t in response], [])
    self.assertEqual([c async for c in response.tool_calls], [])
    self.assertEqual(await response.text(), "")

  async def test_structured_output_lazy_accessor(self):
    """Verifies structured_output resolves the stream and fetches parsed payload from conversation."""
    t_text = types.Text(step_index=1, text="finished")

    async def mock_stream():
      yield t_text

    mock_conv = mock.MagicMock(spec=conversation.Conversation)
    mock_conv.get_last_structured_output.return_value = {"result": "data"}

    response = types.ChatResponse(mock_stream(), conversation=mock_conv)

    data = await response.structured_output()
    self.assertEqual(data, {"result": "data"})
    mock_conv.get_last_structured_output.assert_called_once()

  async def test_usage_metadata_lazy_accessor(self):
    """Verifies usage_metadata resolves the stream and fetches payload from conversation."""
    t_text = types.Text(step_index=1, text="finished")

    async def mock_stream():
      yield t_text

    mock_conv = mock.MagicMock(spec=conversation.Conversation)
    mock_conv.last_turn_usage = types.UsageMetadata(
        prompt_token_count=10,
        candidates_token_count=20,
        total_token_count=30,
    )
    mock_conv._last_turn_usage = mock_conv.last_turn_usage

    response = types.ChatResponse(mock_stream(), conversation=mock_conv)

    await response.resolve()
    usage = response.usage_metadata
    self.assertIsNotNone(usage)
    self.assertEqual(usage.prompt_token_count, 10)
    self.assertEqual(usage.candidates_token_count, 20)
    self.assertEqual(usage.total_token_count, 30)

  async def test_cancel_delegates_to_conversation(self):
    """Verifies that cancel() delegates to conversation.cancel() if not done."""
    async def mock_stream() -> AsyncIterator[types.StreamChunk]:
      yield types.Text(step_index=1, text="hello")
      # Keep it open by not ending or throwing yet

    mock_conv = mock.AsyncMock(spec=conversation.Conversation)
    response = types.ChatResponse(mock_stream(), conversation=mock_conv)

    self.assertFalse(response._is_done)

    await response.cancel()
    mock_conv.cancel.assert_called_once()

  async def test_cancel_completed_stream_is_noop(self):
    """Verifies that cancel() is a safe no-op on a completed stream."""
    async def mock_stream() -> AsyncIterator[types.StreamChunk]:
      yield types.Text(step_index=1, text="hello")

    mock_conv = mock.AsyncMock(spec=conversation.Conversation)
    response = types.ChatResponse(mock_stream(), conversation=mock_conv)

    # Resolve the stream to complete it
    await response.resolve()
    self.assertTrue(response._is_done)

    await response.cancel()
    mock_conv.cancel.assert_not_called()

  async def test_stream_cancellation_sets_is_done(self):
    """Verifies that if stream raises CancelledError, _is_done becomes True."""
    async def mock_cancelled_stream() -> AsyncIterator[types.StreamChunk]:
      yield types.Text(step_index=1, text="hello")
      raise asyncio.CancelledError("Cancelled natively")

    mock_conv = mock.AsyncMock(spec=conversation.Conversation)
    response = types.ChatResponse(
        mock_cancelled_stream(), conversation=mock_conv
    )

    chunks = []
    with self.assertRaisesRegex(asyncio.CancelledError, "Cancelled natively"):
      async for chunk in response.chunks:
        chunks.append(chunk)

    self.assertEqual(len(chunks), 1)
    self.assertTrue(response._is_done)
    self.assertIsInstance(response._stream_error, asyncio.CancelledError)

  def test_thought_chunk_validation(self):
    """Verifies that the Thought subclass validates Pydantic schemas correctly."""
    thought = types.Thought(step_index=1, text="reasoning", signature=b"sig")
    self.assertEqual(thought.step_index, 1)
    self.assertEqual(thought.text, "reasoning")
    self.assertEqual(thought.signature, b"sig")

  def test_text_chunk_validation(self):
    """Verifies that the Text subclass validates Pydantic schemas correctly."""
    text = types.Text(step_index=2, text="conversational answer")
    self.assertEqual(text.step_index, 2)
    self.assertEqual(text.text, "conversational answer")

  async def test_empty_stream_text(self):
    """Verifies that an empty stream resolves text to an empty string."""

    async def mock_empty_stream():
      return
      yield

    response = types.ChatResponse(
        mock_empty_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    self.assertEqual(await response.text(), "")


class McpServerConfigTest(parameterized.TestCase):
  """Validates the McpServerConfig Pydantic models and required fields."""

  @parameterized.named_parameters(
      (
          "stdio",
          types.McpStdioServer,
          {"name": "stdio_server", "command": "node", "args": ["index.js"]},
          {"name": "stdio_server", "command": "node", "args": ["index.js"]},
      ),
      (
          "stdio_with_env",
          types.McpStdioServer,
          {
              "name": "stdio_server",
              "command": "node",
              "args": ["index.js"],
              "env": {"FOO": "bar"},
          },
          {
              "name": "stdio_server",
              "command": "node",
              "args": ["index.js"],
              "env": {"FOO": "bar"},
          },
      ),
      (
          "http",
          types.McpStreamableHttpServer,
          {"name": "http_server", "url": "http://localhost/http"},
          {"name": "http_server", "url": "http://localhost/http"},
      ),
  )
  def test_server_construction_success(
      self, server_cls, init_kwargs, expected_attrs
  ):
    """Verifies that MCP server configs construct successfully with valid arguments.

    What: Checks basic field assignment with required name and command/url.
    Why: Validates the happy path for all MCP connection types.
    How: Constructs the target class with arguments and asserts its properties.

    Args:
      server_cls: The MCP server class under test.
      init_kwargs: Dictionary of constructor arguments.
      expected_attrs: Dict of expected attribute values after construction.
    """
    server = server_cls(**init_kwargs)
    for attr, val in expected_attrs.items():
      self.assertEqual(getattr(server, attr), val)

  @parameterized.named_parameters(
      ("stdio", types.McpStdioServer, {"command": "node"}),
      ("http", types.McpStreamableHttpServer, {"url": "http://localhost/http"}),
  )
  def test_server_missing_name_raises(self, server_cls, init_kwargs):
    """Verifies that MCP server configs raise ValidationError when name is omitted.

    What: Checks required field validation for name.
    Why: Enforces mandatory naming constraints across all server types.
    How: Attempts construction without name and asserts ValidationError.

    Args:
      server_cls: The MCP server class under test.
      init_kwargs: Dictionary of constructor arguments missing the name field.
    """
    with self.assertRaises(pydantic.ValidationError):
      server_cls(**init_kwargs)  # type: ignore

  @parameterized.named_parameters(
      (
          "stdio_enabled",
          types.McpStdioServer,
          {
              "name": "stdio_server",
              "command": "node",
              "enabled_tools": ["tool1"],
          },
          "enabled_tools",
          ["tool1"],
      ),
      (
          "stdio_disabled",
          types.McpStdioServer,
          {
              "name": "stdio_server",
              "command": "node",
              "disabled_tools": ["tool2"],
          },
          "disabled_tools",
          ["tool2"],
      ),
      (
          "http_enabled",
          types.McpStreamableHttpServer,
          {
              "name": "http_server",
              "url": "http://localhost/http",
              "enabled_tools": ["tool1"],
          },
          "enabled_tools",
          ["tool1"],
      ),
      (
          "http_disabled",
          types.McpStreamableHttpServer,
          {
              "name": "http_server",
              "url": "http://localhost/http",
              "disabled_tools": ["tool2"],
          },
          "disabled_tools",
          ["tool2"],
      ),
  )
  def test_server_construction_with_filtering(
      self, server_cls, init_kwargs, expected_attr, expected_val
  ):
    """Verifies that MCP server configs construct with enabled or disabled tools."""
    server = server_cls(**init_kwargs)
    self.assertEqual(getattr(server, expected_attr), expected_val)

  @parameterized.named_parameters(
      (
          "stdio_different_tools",
          types.McpStdioServer,
          {
              "name": "stdio_server",
              "command": "node",
              "enabled_tools": ["tool1"],
              "disabled_tools": ["tool2"],
          },
      ),
      (
          "stdio_same_tool",
          types.McpStdioServer,
          {
              "name": "stdio_server",
              "command": "node",
              "enabled_tools": ["tool1"],
              "disabled_tools": ["tool1"],
          },
      ),
      (
          "http_different_tools",
          types.McpStreamableHttpServer,
          {
              "name": "http_server",
              "url": "http://localhost/http",
              "enabled_tools": ["tool1"],
              "disabled_tools": ["tool2"],
          },
      ),
      (
          "http_same_tool",
          types.McpStreamableHttpServer,
          {
              "name": "http_server",
              "url": "http://localhost/http",
              "enabled_tools": ["tool1"],
              "disabled_tools": ["tool1"],
          },
      ),
  )
  def test_server_exclusive_filtering_raises(self, server_cls, init_kwargs):
    """Verifies that providing both enabled and disabled tools raises ValidationError."""
    with self.assertRaises(pydantic.ValidationError) as ctx:
      server_cls(**init_kwargs)
    self.assertIn(
        "enabled_tools and disabled_tools should be mutually exclusive.",
        str(ctx.exception),
    )

  def test_union_deserialization(self):
    """Verifies that TypeAdapter parses McpServerConfig union variants with name correctly.

    What: Checks Pydantic union polymorphic deserialization.
    Why: Ensures config files or dicts can be parsed polymorphically.
    How: Validates a stdio dict with name against McpServerConfig TypeAdapter.
    """
    adapter = pydantic.TypeAdapter(types.McpServerConfig)

    data = {"type": "stdio", "name": "stdio_server", "command": "node"}
    server = adapter.validate_python(data)
    self.assertIsInstance(server, types.McpStdioServer)
    self.assertEqual(server.name, "stdio_server")

  @parameterized.parameters(
      "validName",
      "valid-name",
      "valid_name",
      "valid_name-123",
  )
  def test_server_valid_names(self, name):
    """Verifies that Gemini-compliant server names pass validation."""
    server = types.McpStdioServer(name=name, command="node")
    self.assertEqual(server.name, name)

  @parameterized.parameters(
      "invalid name",
      "invalid.name",
      "invalid/name",
      "invalid@name",
      "invalidName!",
  )
  def test_server_invalid_names_raise(self, name):
    """Verifies that names violating Gemini's regex pattern trigger validation errors."""
    with self.assertRaises(pydantic.ValidationError):
      types.McpStdioServer(name=name, command="node")


class SubagentCapabilitiesTest(unittest.TestCase):
  """Validates the SubagentCapabilities Pydantic model."""

  def test_defaults(self):
    sc = types.SubagentCapabilities()
    self.assertEqual(sc.agent_behavior, types.AgentBehavior.AUTONOMOUS)
    self.assertIsNone(sc.enabled_tools)
    self.assertIsNone(sc.disabled_tools)

  def test_agent_behavior_explicit(self):
    sc = types.SubagentCapabilities(
        agent_behavior=types.AgentBehavior.INTERACTIVE
    )
    self.assertEqual(sc.agent_behavior, types.AgentBehavior.INTERACTIVE)
    sc_str = types.SubagentCapabilities(agent_behavior="interactive")
    self.assertEqual(sc_str.agent_behavior, types.AgentBehavior.INTERACTIVE)

  def test_mutually_exclusive_ok_enabled(self):
    sc = types.SubagentCapabilities(
        enabled_tools=[types.BuiltinTools.EDIT_FILE]
    )
    self.assertEqual(sc.enabled_tools, [types.BuiltinTools.EDIT_FILE])
    self.assertIsNone(sc.disabled_tools)

  def test_mutually_exclusive_ok_disabled(self):
    sc = types.SubagentCapabilities(
        disabled_tools=[types.BuiltinTools.RUN_COMMAND]
    )
    self.assertIsNone(sc.enabled_tools)
    self.assertEqual(sc.disabled_tools, [types.BuiltinTools.RUN_COMMAND])

  def test_mutually_exclusive_raises(self):
    with self.assertRaises(pydantic.ValidationError):
      types.SubagentCapabilities(
          enabled_tools=[types.BuiltinTools.EDIT_FILE],
          disabled_tools=[types.BuiltinTools.RUN_COMMAND],
      )

  def test_allowed_subagents_valid_when_start_subagent_enabled(self):
    sc = types.SubagentCapabilities(
        enabled_tools=[types.BuiltinTools.START_SUBAGENT],
        allowed_subagents=["worker"],
    )
    self.assertEqual(sc.allowed_subagents, ["worker"])

  def test_allowed_subagents_raises_when_start_subagent_omitted_from_enabled_tools(
      self,
  ):
    with self.assertRaisesRegex(
        pydantic.ValidationError, "START_SUBAGENT is disabled or omitted"
    ):
      types.SubagentCapabilities(
          enabled_tools=[types.BuiltinTools.VIEW_FILE],
          allowed_subagents=["worker"],
      )

  def test_allowed_subagents_raises_when_start_subagent_in_disabled_tools(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError, "START_SUBAGENT is disabled or omitted"
    ):
      types.SubagentCapabilities(
          disabled_tools=[types.BuiltinTools.START_SUBAGENT],
          allowed_subagents=["worker"],
      )


class SubagentConfigTest(unittest.TestCase):
  """Validates the SubagentConfig Pydantic model."""

  def test_basic_construction(self):
    sub = types.SubagentConfig(
        name="helper",
        description="helpful agent",
        system_instructions="always help",
    )
    self.assertEqual(sub.name, "helper")
    self.assertEqual(sub.description, "helpful agent")
    self.assertEqual(sub.system_instructions, "always help")
    self.assertIsNone(sub.capabilities)
    self.assertEqual(sub.tools, [])

  def test_custom_system_instructions(self):
    custom_si = types.CustomSystemInstructions(text="Full custom prompt")
    sub = types.SubagentConfig(
        name="custom_helper",
        description="custom helper agent",
        system_instructions=custom_si,
    )
    self.assertEqual(sub.system_instructions, custom_si)

  def test_templated_system_instructions(self):
    templated_si = types.TemplatedSystemInstructions(
        identity="Helper identity",
        sections=[types.SystemInstructionSection(content="Section 1")],
    )
    sub = types.SubagentConfig(
        name="templated_helper",
        description="templated helper agent",
        system_instructions=templated_si,
    )
    self.assertEqual(sub.system_instructions, templated_si)

  def test_required_fields(self):
    with self.assertRaises(pydantic.ValidationError):
      types.SubagentConfig(**{"name": "helper"})  # Missing description

    with self.assertRaises(pydantic.ValidationError):
      types.SubagentConfig(**{"description": "helpful agent"})  # Missing name


class UsageMetadataTest(unittest.TestCase):
  """Tests for the UsageMetadata class."""

  def test_add_operator(self):
    """Verifies that __add__ sums token usage fields correctly."""
    u1 = types.UsageMetadata(
        prompt_token_count=100,
        cached_content_token_count=50,
        candidates_token_count=30,
        thoughts_token_count=20,
        total_token_count=150,
    )
    u2 = types.UsageMetadata(
        prompt_token_count=200,
        cached_content_token_count=10,
        candidates_token_count=40,
        thoughts_token_count=5,
        total_token_count=245,
    )
    res = u1 + u2
    self.assertEqual(res.prompt_token_count, 300)
    self.assertEqual(res.cached_content_token_count, 60)
    self.assertEqual(res.candidates_token_count, 70)
    self.assertEqual(res.thoughts_token_count, 25)
    self.assertEqual(res.total_token_count, 395)

  def test_add_operator_with_none(self):
    """Verifies that __add__ treats None fields as zero."""
    u1 = types.UsageMetadata(
        prompt_token_count=100,
    )
    u2 = types.UsageMetadata(
        candidates_token_count=50,
    )
    res = u1 + u2
    self.assertEqual(res.prompt_token_count, 100)
    self.assertEqual(res.cached_content_token_count, 0)
    self.assertEqual(res.candidates_token_count, 50)
    self.assertEqual(res.thoughts_token_count, 0)
    self.assertEqual(res.total_token_count, 0)

  def test_add_operator_service_tier(self):
    """Verifies that __add__ merges service_tier commutatively."""
    u_none = types.UsageMetadata()
    u_std = types.UsageMetadata(service_tier=types.ServiceTier.STANDARD)
    u_pri = types.UsageMetadata(service_tier=types.ServiceTier.PRIORITY)
    u_flex = types.UsageMetadata(service_tier=types.ServiceTier.FLEX)

    self.assertEqual((u_pri + u_pri).service_tier, types.ServiceTier.PRIORITY)
    self.assertEqual((u_flex + u_flex).service_tier, types.ServiceTier.FLEX)
    self.assertEqual((u_pri + u_none).service_tier, types.ServiceTier.PRIORITY)
    self.assertEqual((u_none + u_flex).service_tier, types.ServiceTier.FLEX)
    self.assertEqual((u_pri + u_flex).service_tier, types.ServiceTier.STANDARD)
    self.assertEqual((u_flex + u_pri).service_tier, types.ServiceTier.STANDARD)
    self.assertEqual((u_pri + u_std).service_tier, types.ServiceTier.STANDARD)

  def test_add_operator_invalid_type(self):
    """Verifies that __add__ returns NotImplemented for invalid types."""
    u = types.UsageMetadata(prompt_token_count=10)
    self.assertEqual(u.__add__(1), NotImplemented)

  def test_sub_operator(self):
    """Verifies that __sub__ subtracts token usage fields correctly."""
    u1 = types.UsageMetadata(
        prompt_token_count=300,
        cached_content_token_count=60,
        candidates_token_count=70,
        thoughts_token_count=25,
        total_token_count=395,
    )
    u2 = types.UsageMetadata(
        prompt_token_count=100,
        cached_content_token_count=50,
        candidates_token_count=30,
        thoughts_token_count=20,
        total_token_count=150,
    )
    res = u1 - u2
    self.assertEqual(res.prompt_token_count, 200)
    self.assertEqual(res.cached_content_token_count, 10)
    self.assertEqual(res.candidates_token_count, 40)
    self.assertEqual(res.thoughts_token_count, 5)
    self.assertEqual(res.total_token_count, 245)

  def test_sub_operator_with_none(self):
    """Verifies that __sub__ treats None fields as zero."""
    u1 = types.UsageMetadata(
        prompt_token_count=100,
    )
    u2 = types.UsageMetadata(
        candidates_token_count=50,
    )
    res = u1 - u2
    self.assertEqual(res.prompt_token_count, 100)
    self.assertEqual(res.cached_content_token_count, 0)
    self.assertEqual(res.candidates_token_count, -50)
    self.assertEqual(res.thoughts_token_count, 0)
    self.assertEqual(res.total_token_count, 0)

  def test_sub_operator_service_tier(self):
    """Verifies that __sub__ preserves and resolves service_tier correctly."""
    u_none = types.UsageMetadata()
    u_std = types.UsageMetadata(service_tier=types.ServiceTier.STANDARD)
    u_pri = types.UsageMetadata(service_tier=types.ServiceTier.PRIORITY)
    u_flex = types.UsageMetadata(service_tier=types.ServiceTier.FLEX)

    self.assertEqual((u_pri - u_pri).service_tier, types.ServiceTier.PRIORITY)
    self.assertEqual((u_flex - u_flex).service_tier, types.ServiceTier.FLEX)
    self.assertEqual((u_pri - u_none).service_tier, types.ServiceTier.PRIORITY)
    self.assertEqual((u_none - u_flex).service_tier, types.ServiceTier.FLEX)
    self.assertEqual((u_pri - u_flex).service_tier, types.ServiceTier.PRIORITY)
    self.assertEqual((u_flex - u_pri).service_tier, types.ServiceTier.FLEX)
    self.assertEqual((u_pri - u_std).service_tier, types.ServiceTier.PRIORITY)
    self.assertIsNone((u_none - u_none).service_tier)

  def test_sub_operator_invalid_type(self):
    """Verifies that __sub__ returns NotImplemented for invalid types."""
    u = types.UsageMetadata(prompt_token_count=10)
    self.assertEqual(u.__sub__(1), NotImplemented)


class RetryConfigTest(unittest.TestCase):
  """Tests for RetryConfig presets and explicit configuration."""

  def test_benchmark_preset(self):
    cfg = types.RetryConfig.benchmark()
    self.assertIsNotNone(cfg.api_retry)
    self.assertEqual(cfg.api_retry.max_retries, 2**32 - 1)
    self.assertEqual(cfg.api_retry.initial_sleep_duration_ms, 1000)
    self.assertIsNone(cfg.model_output_retry)

  def test_explicit_models(self):
    cfg = types.RetryConfig(
        api_retry=types.ModelAPIRetryConfig(max_retries=5),
        model_output_retry=types.ModelOutputRetryConfig(max_retries=3),
    )
    self.assertEqual(cfg.api_retry.max_retries, 5)
    self.assertEqual(cfg.model_output_retry.max_retries, 3)

  def test_uint32_validation(self):
    with self.assertRaises(pydantic.ValidationError):
      types.ModelAPIRetryConfig(max_retries=-1)
    with self.assertRaises(pydantic.ValidationError):
      types.ModelAPIRetryConfig(max_retries=2**32)
    with self.assertRaises(pydantic.ValidationError):
      types.ModelOutputRetryConfig(max_retries=-5)
    with self.assertRaises(pydantic.ValidationError):
      types.ModelOutputRetryConfig(max_retries=2**32)


class BudgetEnforcementTypesTest(absltest.TestCase):
  """Tests for StopReason."""

  def test_budget_config_defaults_and_validation(self):
    cfg = types.BudgetConfig()
    self.assertIsNone(cfg.max_model_calls)
    self.assertIsNone(cfg.max_tool_calls)
    self.assertIsNone(cfg.max_input_tokens)
    self.assertIsNone(cfg.max_output_tokens)
    self.assertIsNone(cfg.max_total_tokens)
    self.assertEqual(cfg.scope, types.BudgetScope.LIFETIME)

    cfg_valid = types.BudgetConfig(
        max_model_calls=5,
        max_tool_calls=10,
        max_input_tokens=500,
        max_output_tokens=200,
        max_total_tokens=1000,
        scope=types.BudgetScope.FORWARD_LOOKING,
    )
    self.assertEqual(cfg_valid.max_model_calls, 5)
    self.assertEqual(cfg_valid.max_tool_calls, 10)
    self.assertEqual(cfg_valid.max_input_tokens, 500)
    self.assertEqual(cfg_valid.max_output_tokens, 200)
    self.assertEqual(cfg_valid.max_total_tokens, 1000)
    self.assertEqual(cfg_valid.scope, types.BudgetScope.FORWARD_LOOKING)

    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_model_calls=0)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_model_calls=2**31)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_tool_calls=0)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_tool_calls=2**31)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_input_tokens=0)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_input_tokens=2**63)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_output_tokens=0)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_output_tokens=2**63)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_total_tokens=0)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig(max_total_tokens=2**63)
    with self.assertRaises(pydantic.ValidationError):
      types.BudgetConfig.model_validate({"scope": "INVALID_SCOPE"})

  def test_budget_scope_enum(self):
    self.assertEqual(types.BudgetScope.LIFETIME, "LIFETIME")
    self.assertEqual(types.BudgetScope.FORWARD_LOOKING, "FORWARD_LOOKING")

  def test_stop_reason_enum(self):
    self.assertEqual(types.StopReason.UNSPECIFIED, "UNSPECIFIED")
    self.assertEqual(
        types.StopReason.MAX_MODEL_CALLS_EXCEEDED,
        "MAX_MODEL_CALLS_EXCEEDED",
    )
    self.assertEqual(
        types.StopReason.MAX_TOOL_CALLS_EXCEEDED,
        "MAX_TOOL_CALLS_EXCEEDED",
    )
    self.assertEqual(
        types.StopReason.MAX_INPUT_TOKENS_EXCEEDED,
        "MAX_INPUT_TOKENS_EXCEEDED",
    )
    self.assertEqual(
        types.StopReason.MAX_OUTPUT_TOKENS_EXCEEDED,
        "MAX_OUTPUT_TOKENS_EXCEEDED",
    )
    self.assertEqual(
        types.StopReason.MAX_TOTAL_TOKENS_EXCEEDED,
        "MAX_TOTAL_TOKENS_EXCEEDED",
    )
    self.assertEqual(types.StopReason.QUOTA_EXHAUSTED, "QUOTA_EXHAUSTED")

  def test_chat_response_stop_reason(self):
    async def mock_stream():
      yield types.Text(step_index=1, text="")

    mock_conv = mock.MagicMock(spec=conversation.Conversation)
    mock_conv._last_turn_stop_reason = types.StopReason.QUOTA_EXHAUSTED
    response = types.ChatResponse(mock_stream(), conversation=mock_conv)
    self.assertEqual(
        response.stop_reason,
        types.StopReason.QUOTA_EXHAUSTED,
    )


class StopHookTypesTest(absltest.TestCase):
  """Tests for StopDecision, StopHookResult, and StopArgs."""

  def test_stop_decision_enum(self):
    self.assertEqual(types.StopDecision.ALLOW_STOP, "ALLOW_STOP")
    self.assertEqual(types.StopDecision.CONTINUE, "CONTINUE")

  def test_stop_hook_result_defaults(self):
    res = types.StopHookResult()
    self.assertEqual(res.decision, types.StopDecision.ALLOW_STOP)
    self.assertEqual(res.reason, "")

  def test_stop_hook_result_custom(self):
    res = types.StopHookResult(
        decision=types.StopDecision.CONTINUE,
        reason="Needs revision",
    )
    self.assertEqual(res.decision, types.StopDecision.CONTINUE)
    self.assertEqual(res.reason, "Needs revision")

  def test_stop_hook_result_extra_fields_ignored(self):
    res = types.StopHookResult.model_validate(
        {"decision": "CONTINUE", "reason": "more work", "extra_field": 123}
    )
    self.assertEqual(res.decision, types.StopDecision.CONTINUE)
    self.assertEqual(res.reason, "more work")

  def test_stop_hook_result_continue_requires_non_empty_reason(self):
    with self.assertRaisesRegex(ValueError, "requires a non-empty reason"):
      types.StopHookResult(decision=types.StopDecision.CONTINUE)

    with self.assertRaisesRegex(ValueError, "requires a non-empty reason"):
      types.StopHookResult(decision=types.StopDecision.CONTINUE, reason="")

    with self.assertRaisesRegex(ValueError, "requires a non-empty reason"):
      types.StopHookResult(decision=types.StopDecision.CONTINUE, reason="   ")

  def test_stop_args_defaults(self):
    args = types.StopArgs()
    self.assertEqual(args.response_text, "")
    self.assertEqual(args.trajectory_id, "")
    self.assertEqual(args.continuation_count, 0)
    self.assertEqual(args.stop_reason, types.StopReason.UNSPECIFIED)
    self.assertEqual(args.error_message, "")

  def test_stop_args_custom(self):
    args = types.StopArgs(
        response_text="All done",
        trajectory_id="traj_123",
        continuation_count=2,
        stop_reason=types.StopReason.MAX_MODEL_CALLS_EXCEEDED,
        error_message="budget reached",
    )
    self.assertEqual(args.response_text, "All done")
    self.assertEqual(args.trajectory_id, "traj_123")
    self.assertEqual(args.continuation_count, 2)
    self.assertEqual(
        args.stop_reason, types.StopReason.MAX_MODEL_CALLS_EXCEEDED
    )
    self.assertEqual(args.error_message, "budget reached")

  def test_stop_args_string_coercion_and_extra_ignored(self):
    args = types.StopArgs.model_validate({
        "response_text": "done",
        "stop_reason": "QUOTA_EXHAUSTED",
        "unknown_wire_field": "ignore_me",
    })
    self.assertEqual(args.response_text, "done")
    self.assertEqual(args.stop_reason, types.StopReason.QUOTA_EXHAUSTED)


if __name__ == "__main__":
  absltest.main()
