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

"""Unit tests for LocalConnection."""

import asyncio
import base64
import datetime
import enum
import importlib
import io
import json
import os
import pathlib
import struct
import subprocess
import tempfile
from typing import Any, Literal, Union
import typing
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pydantic
import websockets

from google.antigravity.proto import localharness_pb2
from google.antigravity import models as models_lib
from google.antigravity import types
from google.antigravity.connections.local import event_processor
from google.antigravity.connections.local import local_connection
from google.antigravity.connections.local import local_connection_config
from google.antigravity.connections.local import test_utils
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks as hooks_base
from google.antigravity.hooks import policy
from google.antigravity.tools import tool_context
from google.antigravity.tools import tool_runner


class PromptSanitizationTest(unittest.TestCase):
  """Tests for _sanitize_prompt and to_proto_input_content."""

  def test_sanitize_prompt_null_bytes_and_control_chars(self):
    sanitized = local_connection._sanitize_prompt("Hello\x00World\x07!\x7f\x80")
    self.assertEqual(sanitized, "Hello World !  ")

  def test_sanitize_prompt_preserves_whitespace(self):
    sanitized = local_connection._sanitize_prompt("Line1\nLine2\r\tTab")
    self.assertEqual(sanitized, "Line1\nLine2\r\tTab")

  def test_sanitize_prompt_empty_or_whitespace_fallback(self):
    self.assertEqual(local_connection._sanitize_prompt(""), "")
    self.assertEqual(local_connection._sanitize_prompt("\x00\x00"), " ")

  def test_to_proto_input_content_sanitizes_strings(self):
    part = local_connection.to_proto_input_content("Bad\x00Input\x7f")
    self.assertEqual(part.text, "Bad Input ")


class CallableToToolProtoTest(unittest.TestCase):
  """Tests for callable_to_tool_proto schema conversion."""

  def test_callable_to_tool_proto_schema_lowercase_types(self):
    def get_temperature(location: str) -> str:
      """Get the current temperature."""
      return f"The temperature in {location} is 72°F."

    proto = local_connection.callable_to_tool_proto(get_temperature)
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")
    self.assertEqual(
        schema.get("properties", {}).get("location", {}).get("type"),
        "string",
    )

  def test_callable_to_tool_proto_complex_types(self):
    class ItemDetail(pydantic.BaseModel):
      item_name: str
      quantity: int = 1

    # pylint: disable=unused-argument
    def create_order(
        order_id: str,
        count: int,
        price: float,
        is_express: bool,
        tags: list[str],
        items: list[ItemDetail],
    ) -> dict[str, str]:
      """Create a new order with details."""
      return {"order_id": order_id}
    # pylint: enable=unused-argument

    proto = local_connection.callable_to_tool_proto(create_order)
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")
    props = schema.get("properties", {})
    self.assertEqual(props.get("order_id", {}).get("type"), "string")
    self.assertEqual(props.get("count", {}).get("type"), "integer")
    self.assertEqual(props.get("price", {}).get("type"), "number")
    self.assertEqual(props.get("is_express", {}).get("type"), "boolean")
    self.assertEqual(props.get("tags", {}).get("type"), "array")
    self.assertEqual(
        props.get("tags", {}).get("items", {}).get("type"), "string"
    )
    self.assertEqual(props.get("items", {}).get("type"), "array")

  def test_callable_to_tool_proto_enum_preserves_values(self):
    class Direction(enum.Enum):
      NORTH = "NORTH"
      SOUTH = "SOUTH"
      EAST = "EAST"
      WEST = "WEST"

    def navigate(direction: Direction) -> str:
      """Navigate in a direction."""
      return f"Heading {direction.value}"

    proto = local_connection.callable_to_tool_proto(navigate)
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")
    dir_prop = schema.get("properties", {}).get("direction", {})
    self.assertEqual(dir_prop.get("type"), "string")
    self.assertEqual(
        dir_prop.get("enum"), ["NORTH", "SOUTH", "EAST", "WEST"]
    )

  def test_callable_to_tool_proto_no_params(self):
    def get_version() -> str:
      """Return system version."""
      return "1.0.0"

    proto = local_connection.callable_to_tool_proto(get_version)
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")

  def test_callable_to_tool_proto_tool_with_schema(self):
    def raw_search(query: str) -> str:
      """Search documents."""
      return query

    tool = tool_runner.ToolWithSchema(
        fn=raw_search,
        input_schema={
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING"}},
        },
    )

    proto = local_connection.callable_to_tool_proto(tool)
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")
    self.assertEqual(
        schema.get("properties", {}).get("query", {}).get("type"), "string"
    )

  def test_callable_to_tool_proto_param_named_type_collision(self):
    # pylint: disable=redefined-builtin
    def configure(type: str, timeout: int) -> dict[str, Any]:
      """Configure system with type and timeout."""
      return {"type": type, "timeout": timeout}
    # pylint: enable=redefined-builtin

    proto = local_connection.callable_to_tool_proto(configure)
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")
    props = schema.get("properties", {})
    self.assertIn("type", props)
    self.assertEqual(props["type"].get("type"), "string")
    self.assertEqual(props.get("timeout", {}).get("type"), "integer")

  def test_callable_to_tool_proto_literal_enum_preservation(self):
    def set_log_level(
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
    ) -> str:
      """Set logging level."""
      return level

    proto = local_connection.callable_to_tool_proto(set_log_level)
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")
    level_prop = schema.get("properties", {}).get("level", {})
    self.assertEqual(level_prop.get("type"), "string")
    self.assertEqual(
        level_prop.get("enum"), ["DEBUG", "INFO", "WARNING", "ERROR"]
    )

  def test_callable_to_tool_proto_union_combiners(self):
    def process_id(identifier: Union[int, str]) -> str:
      """Process an identifier that can be int or str."""
      return str(identifier)

    proto = local_connection.callable_to_tool_proto(process_id)
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")
    id_prop = schema.get("properties", {}).get("identifier", {})
    self.assertIn("anyOf", id_prop)
    self.assertNotIn("any_of", id_prop)
    combiner = id_prop["anyOf"]
    types_in_union = {c.get("type") for c in combiner if isinstance(c, dict)}
    self.assertTrue({"integer", "string"}.issubset(types_in_union))

  def test_callable_to_tool_proto_tool_context_injection_stripping(self):
    # pylint: disable=unused-argument
    def inspect_tool(ctx: tool_context.ToolContext, query: str) -> str:
      """Tool that requests context injection."""
      return query
    # pylint: enable=unused-argument

    runner = tool_runner.ToolRunner([inspect_tool])
    proto = local_connection.callable_to_tool_proto(
        inspect_tool, tool_runner=runner
    )
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")
    props = schema.get("properties", {})
    self.assertNotIn("ctx", props)
    self.assertIn("query", props)
    self.assertEqual(props["query"].get("type"), "string")

  def test_callable_to_tool_proto_tool_with_schema_uppercase_types(self):
    def raw_fn(x: int) -> int:
      return x

    tool = tool_runner.ToolWithSchema(
        fn=raw_fn,
        input_schema={
            "type": "OBJECT",
            "properties": {
                "count": {"type": "INTEGER"},
                "rate": {"type": "NUMBER"},
            },
            "const": "UPPERCASE_CONST",
            "default": "DEFAULT_VAL",
        },
    )
    proto = local_connection.callable_to_tool_proto(tool)
    schema = json.loads(proto.parameters_json_schema)
    self.assertEqual(schema.get("type"), "object")
    self.assertEqual(
        schema.get("properties", {}).get("count", {}).get("type"), "integer"
    )
    self.assertEqual(
        schema.get("properties", {}).get("rate", {}).get("type"), "number"
    )
    self.assertEqual(schema.get("const"), "UPPERCASE_CONST")
    self.assertEqual(schema.get("default"), "DEFAULT_VAL")


class LocalConnectionTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock(spec=subprocess.Popen)
    self.tool_runner = tool_runner.ToolRunner()

  def _make_harness(self, hook_runner=None, initial_history=None):
    return test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hook_runner,
        initial_history=initial_history,
    )

  async def test_initial_handshake_synchronization(self):
    """Verifies that L2 connections receive restored historical steps synchronously during initialization and start fully idle."""
    hist_step = local_connection.LocalConnectionStep(
        step_index=1,
        content="Historical text",
        status=types.StepStatus.DONE,
        source=types.StepSource.MODEL,
    )
    harness = self._make_harness(initial_history=[hist_step])

    # 1. Confirm connection starts 100% fully idle by default
    self.assertTrue(harness.conn.is_idle)

    # 2. Confirm historical steps are exposed immediately on _initial_history
    self.assertEqual(len(harness.conn._initial_history), 1)
    self.assertEqual(
        harness.conn._initial_history[0].content, "Historical text"
    )

  async def test_receive_steps_basic(self):
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            text="Hello world",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()

    # Simulate that a turn is active (send clears this in reality)
    harness.conn._is_idle.clear()

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "Hello world")
    self.assertEqual(steps[0].status, types.StepStatus.ACTIVE)
    self.assertEqual(steps[0].source, types.StepSource.MODEL)

  async def test_receive_steps_system_error(self):
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            error=localharness_pb2.ActionError(
                error_message="Fatal system failure",
                http_code=400,
            ),
            state=localharness_pb2.StepUpdate.STATE_ERROR,
            source=localharness_pb2.StepUpdate.SOURCE_SYSTEM,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()
    harness.conn._is_idle.clear()

    # receive_steps should raise AntigravityConnectionError when it
    # encounters the system error step.
    with self.assertRaisesRegex(
        types.AntigravityConnectionError, "Fatal system failure"
    ):
      async for _ in harness.conn.receive_steps():
        pass

  async def test_receive_steps_system_error_401(self):
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            error=localharness_pb2.ActionError(
                error_message="Unauthorized access",
                http_code=401,
            ),
            state=localharness_pb2.StepUpdate.STATE_ERROR,
            source=localharness_pb2.StepUpdate.SOURCE_SYSTEM,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()
    harness.conn._is_idle.clear()

    # receive_steps should raise AntigravityConnectionError when it
    # encounters the system error step.
    with self.assertRaisesRegex(
        types.AntigravityConnectionError, "Unauthorized access"
    ):
      async for _ in harness.conn.receive_steps():
        pass

  async def test_receive_steps_system_error_429(self):
    harness = self._make_harness()
    mock_trajectory_id = "my_cascade"

    await harness.conn.send("Hello")
    init_data = await harness.wait_for_response()
    self.assertEqual(init_data.get("userInput"), {"parts": [{"text": "Hello"}]})

    # Set the cascade ID and send the 429 error step.
    event1 = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id=mock_trajectory_id,
            trajectory_id=mock_trajectory_id,
            step_index=1,
            error=localharness_pb2.ActionError(
                error_message="Resource exhausted",
                http_code=429,
            ),
            state=localharness_pb2.StepUpdate.STATE_ERROR,
            source=localharness_pb2.StepUpdate.SOURCE_SYSTEM,
        )
    )
    await harness.send_event(event1)

    # Send idle update with error.
    event2 = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id=mock_trajectory_id,
            state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
            error="executor run failed: Resource exhausted",
        )
    )
    await harness.send_event(event2)
    await harness.close_from_harness_side()

    steps = []
    with self.assertRaisesRegex(
        types.AntigravityExecutionError,
        "executor run failed: Resource exhausted",
    ):
      async for step in harness.conn.receive_steps():
        steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].error, "Resource exhausted")
    self.assertEqual(steps[0].status, types.StepStatus.ERROR)

  async def test_receive_steps_trajectory_error(self):
    harness = self._make_harness()

    await harness.conn.send("Hello")
    init_data = await harness.wait_for_response()
    self.assertEqual(init_data.get("userInput"), {"parts": [{"text": "Hello"}]})

    # Set the cascade ID.
    event1 = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="my_cascade",
            trajectory_id="my_cascade",
            step_index=1,
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            text="I'm working",
        )
    )
    await harness.send_event(event1)

    # Send an error.
    event2 = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="my_cascade",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
            error="Trajectory execution failed",
        )
    )
    await harness.send_event(event2)
    await harness.close_from_harness_side()

    steps = []
    with self.assertRaisesRegex(
        types.AntigravityExecutionError, "Trajectory execution failed"
    ):
      async for step in harness.conn.receive_steps():
        steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "I'm working")

  async def test_receive_steps_mcp_load_failure(self):
    harness = self._make_harness()

    await harness.conn.send("Hello")
    init_data = await harness.wait_for_response()
    self.assertEqual(init_data.get("userInput"), {"parts": [{"text": "Hello"}]})

    # Send an error indicating MCP failure.
    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="my_cascade",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
            error="MCP load failed for dead_server: connection refused",
        )
    )
    await harness.send_event(event)
    await harness.close_from_harness_side()

    with self.assertRaisesRegex(
        types.AntigravityExecutionError,
        "MCP load failed for dead_server: connection refused",
    ):
      async for _ in harness.conn.receive_steps():
        pass

  async def test_receive_steps_multiple_idle_state_updates_hang(self):
    """Verifies receive_steps does not hang on multiple STATE_IDLE events."""
    harness = self._make_harness()

    await harness.conn.send("Hello")
    init_data = await harness.wait_for_response()
    self.assertEqual(init_data.get("userInput"), {"parts": [{"text": "Hello"}]})

    # 1. Harness sends STATE_IDLE (e.g. while processing tool call)
    event1 = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="my_cascade",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
        )
    )
    await harness.send_event(event1)

    # 2. Additional step arrives after first STATE_IDLE
    event2 = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            trajectory_id="my_cascade",
            step_index=1,
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            text="Step content after idle",
        )
    )
    await harness.send_event(event2)

    # 3. Final STATE_IDLE for turn completion
    event3 = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="my_cascade",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
        )
    )
    await harness.send_event(event3)

    steps = []
    async def _collect():
      async for step in harness.conn.receive_steps():
        steps.append(step)

    await asyncio.wait_for(_collect(), timeout=1.0)
    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "Step content after idle")

  def test_local_connection_step_from_dict(self):
    """Tests that LocalConnectionStep maps fields correctly."""
    step_dict = {
        "step_index": 1,
        "text": "Hello world",
        "state": "STATE_ACTIVE",
        "source": "SOURCE_MODEL",
        "target": "TARGET_USER",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.id, "1")
    self.assertEqual(step.content, "Hello world")
    self.assertEqual(step.status, types.StepStatus.ACTIVE)
    self.assertEqual(step.source, types.StepSource.MODEL)
    self.assertEqual(step.target, "TARGET_USER")

  def test_local_connection_step_from_dict_thinking(self):
    """Tests that thinking field is correctly populated from step dict."""
    step_dict = {
        "step_index": 1,
        "text": "",
        "thinking": "Let me analyze this step by step.",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.thinking, "Let me analyze this step by step.")
    self.assertEqual(step.content, "")

  def test_local_connection_step_from_dict_thinking_empty_by_default(self):
    """Tests that thinking defaults to empty string when not present."""
    step_dict = {
        "step_index": 1,
        "text": "Hello",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.thinking, "")
    self.assertEqual(step.content, "Hello")

  async def test_receive_steps_thinking_populated(self):
    """Tests that thinking field flows from proto through to SDK Step."""
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            text="",
            thinking="Internal reasoning about the problem.",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()
    harness.conn._is_idle.clear()

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].thinking, "Internal reasoning about the problem.")
    self.assertEqual(steps[0].content, "")

  async def test_receive_steps_thinking_and_text_independent(self):
    """Tests that thinking and text are independent, non-exclusive fields.

    This is the key behavioral invariant: the translator must populate both
    fields from the same model response. A regression to mutually exclusive
    branches would zero out one of the two.
    """
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            text="Here is my answer.",
            thinking="Let me reason through this carefully.",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()
    harness.conn._is_idle.clear()

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "Here is my answer.")
    self.assertEqual(steps[0].thinking, "Let me reason through this carefully.")

  async def test_thinking_only_step_is_target_user_not_complete(self):
    """Tests that thinking-only steps are TARGET_USER but not is_complete_response.

    Thinking is user-visible output (TARGET_USER), but a step with only
    thinking and no text must not be flagged as a complete response —
    otherwise the SDK would prematurely treat the turn as finished.
    """
    step_dict = {
        "step_index": 1,
        "text": "",
        "thinking": "Internal reasoning about the problem.",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
        "target": "TARGET_USER",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.thinking, "Internal reasoning about the problem.")
    self.assertEqual(step.target, "TARGET_USER")
    self.assertFalse(step.is_complete_response)

  def test_local_connection_step_from_dict_content_delta(self):
    """Tests that content_delta is correctly parsed from text_delta."""
    step_dict = {
        "step_index": 1,
        "text": "Hello world",
        "text_delta": " world",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.content, "Hello world")
    self.assertEqual(step.content_delta, " world")

  def test_local_connection_step_from_dict_thinking_delta(self):
    """Tests that thinking_delta is correctly parsed."""
    step_dict = {
        "step_index": 1,
        "text": "",
        "thinking": "Step 1. Step 2.",
        "thinking_delta": " Step 2.",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.thinking, "Step 1. Step 2.")
    self.assertEqual(step.thinking_delta, " Step 2.")

  def test_local_connection_step_from_dict_deltas_default_empty(self):
    """Tests that delta fields default to empty when not present."""
    step_dict = {
        "step_index": 1,
        "text": "Hello",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.content_delta, "")
    self.assertEqual(step.thinking_delta, "")

  async def test_turn_hook_deny(self):
    hr = hook_runner.HookRunner()

    @hooks_base.pre_turn
    async def denying_turn(_):
      return hooks_base.HookResult(allow=False, message="Denied by hook")

    hr.register_hook(denying_turn)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await harness.conn.send("Hello")
    await harness.send_event(
        localharness_pb2.OutputEvent(
            call_hook_request=localharness_pb2.CallHookRequest(
                request_id="req_deny",
                name="PreTurn",
                type=localharness_pb2.LIFECYCLE_HOOK_PRE_TURN,
                pre_turn_args=localharness_pb2.PreTurnArgs(
                    user_input=localharness_pb2.UserInput(
                        parts=[localharness_pb2.UserInput.Part(text="Hello")]
                    )
                ),
            )
        )
    )

    # The harness emits STATE_CANCELLED when the PreTurn hook denies.
    await harness.send_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="test",
                state=localharness_pb2.TrajectoryStateUpdate.State.STATE_CANCELLED,
                error="Denied by hook",
            )
        )
    )

    with self.assertRaises(types.AntigravityExecutionError) as ctx:
      async for _ in harness.conn.receive_steps():
        pass

    self.assertIn("Denied by hook", str(ctx.exception))

  async def test_send_none_dispatches_turn_hook_with_empty_string(self):
    hr = hook_runner.HookRunner()
    captured = []

    @hooks_base.pre_turn
    async def capturing_turn(data: str) -> hooks_base.HookResult:
      captured.append(data)
      return hooks_base.HookResult(allow=True)

    hr.register_hook(capturing_turn)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await harness.conn.send(None)
    await harness.send_event(
        localharness_pb2.OutputEvent(
            call_hook_request=localharness_pb2.CallHookRequest(
                request_id="req_none",
                name="PreTurn",
                type=localharness_pb2.LIFECYCLE_HOOK_PRE_TURN,
                pre_turn_args=localharness_pb2.PreTurnArgs(
                    user_input=localharness_pb2.UserInput(
                        parts=[localharness_pb2.UserInput.Part(text="")]
                    )
                ),
            )
        )
    )
    await asyncio.sleep(0.05)
    self.assertEqual(captured, [""])

  async def test_tool_hooks_receive_call_id(self):
    hr = hook_runner.HookRunner()
    pre_ids = []
    post_ids = []
    error_ids = []

    @hooks_base.pre_tool_call_decide
    async def capture_pre(data: types.ToolCall) -> hooks_base.HookResult:
      pre_ids.append(data.id)
      return hooks_base.HookResult(allow=True)

    @hooks_base.post_tool_call
    async def capture_post(data: types.ToolResult) -> None:
      post_ids.append(data.id)

    @hooks_base.on_tool_error
    async def capture_error(data: Exception) -> None:
      error_ids.append(getattr(data, "call_id", None))
      return None

    hr.register_hook(capture_pre)
    hr.register_hook(capture_post)
    hr.register_hook(capture_error)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    # Simulate harness dispatching PreTool, PostTool, and OnToolError hooks
    await harness.send_event(
        localharness_pb2.OutputEvent(
            call_hook_request=localharness_pb2.CallHookRequest(
                request_id="req_pre",
                name="PreTool",
                type=localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL,
                pre_tool_args=localharness_pb2.PreToolArgs(
                    tool_name="greet",
                    arguments_json='{"name": "Alice"}',
                    call_id="call_pre_123",
                ),
            )
        )
    )
    await harness.send_event(
        localharness_pb2.OutputEvent(
            call_hook_request=localharness_pb2.CallHookRequest(
                request_id="req_post",
                name="PostTool",
                type=localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
                post_tool_args=localharness_pb2.PostToolArgs(
                    tool_name="greet",
                    result="Hello Alice",
                    call_id="call_post_456",
                ),
            )
        )
    )
    await harness.send_event(
        localharness_pb2.OutputEvent(
            call_hook_request=localharness_pb2.CallHookRequest(
                request_id="req_err",
                name="OnToolError",
                type=localharness_pb2.LIFECYCLE_HOOK_ON_TOOL_ERROR,
                on_tool_error_args=localharness_pb2.OnToolErrorArgs(
                    tool_name="broken",
                    error_message="failed",
                    call_id="call_err_789",
                ),
            )
        )
    )
    await asyncio.sleep(0.05)
    self.assertEqual(pre_ids, ["call_pre_123"])
    self.assertEqual(post_ids, ["call_post_456"])
    self.assertEqual(error_ids, ["call_err_789"])

  def test_extract_media_from_result(self):
    img = types.Image(data=b"\xff\xd8\xff\xd9", mime_type="image/jpeg")

    # A bare media value is fully extracted.
    cleaned, media = event_processor._extract_media_from_result(img)
    self.assertIsNone(cleaned)
    self.assertEqual(media, [img])

    # Media is pulled out of a mixed list, text is kept.
    cleaned, media = event_processor._extract_media_from_result(
        ["a", img, "b"]
    )
    self.assertEqual(cleaned, ["a", "b"])
    self.assertEqual(media, [img])

    # Non-media values pass through untouched.
    cleaned, media = event_processor._extract_media_from_result({"k": "v"})
    self.assertEqual(cleaned, {"k": "v"})
    self.assertEqual(media, [])

    # Media nested in a dict is extracted; remaining keys are kept.
    cleaned, media = event_processor._extract_media_from_result(
        {"caption": "hi", "img": img}
    )
    self.assertEqual(cleaned, {"caption": "hi"})
    self.assertEqual(media, [img])

  async def test_tool_result_image_sent_as_supplemental_media(self):
    def image_tool():
      return [
          "here is the snapshot",
          types.Image(data=b"\xff\xd8\xff\xd9", mime_type="image/jpeg"),
      ]

    self.tool_runner.register(image_tool, name="image_tool")
    harness = self._make_harness()

    await harness.send_event(
        localharness_pb2.OutputEvent(
            tool_call=localharness_pb2.ToolCall(
                id="call_img", name="image_tool", arguments_json="{}"
            )
        )
    )

    sent_data = await harness.wait_for_response()
    resp = sent_data["toolResponse"]
    self.assertEqual(resp["id"], "call_img")
    # The text part stays in response_json; the image becomes supplemental
    # media.
    self.assertIn("here is the snapshot", resp["responseJson"])
    self.assertIn("supplementalMedia", resp)
    self.assertEqual(resp["supplementalMedia"][0]["mimeType"], "image/jpeg")
    self.assertEqual(
        base64.b64decode(resp["supplementalMedia"][0]["data"]),
        b"\xff\xd8\xff\xd9",
    )

  async def test_tool_result_media_only_uses_placeholder_and_description(self):
    def photo_tool():
      # Returns ONLY media (no accompanying text), with a description.
      return types.Image(
          data=b"\xff\xd8\xff\xd9",
          mime_type="image/jpeg",
          description="a deck photo",
      )

    self.tool_runner.register(photo_tool, name="photo_tool")
    harness = self._make_harness()

    await harness.send_event(
        localharness_pb2.OutputEvent(
            tool_call=localharness_pb2.ToolCall(
                id="call_photo", name="photo_tool", arguments_json="{}"
            )
        )
    )

    sent_data = await harness.wait_for_response()
    resp = sent_data["toolResponse"]
    self.assertEqual(resp["id"], "call_photo")
    # A media-only result gets a placeholder text result, and the image is
    # carried as supplemental media (with its description preserved).
    self.assertIn("Returned 1 media attachment(s)", resp["responseJson"])
    self.assertEqual(
        resp["supplementalMedia"][0]["description"], "a deck photo"
    )
    self.assertEqual(
        base64.b64decode(resp["supplementalMedia"][0]["data"]),
        b"\xff\xd8\xff\xd9",
    )

  async def test_large_tool_result_handled(self):
    def large_tool():
      return "X" * (5 * 1024 * 1024)

    self.tool_runner.register(large_tool, name="large_tool")
    harness = self._make_harness()

    await harness.send_event(
        localharness_pb2.OutputEvent(
            tool_call=localharness_pb2.ToolCall(
                id="call_large", name="large_tool", arguments_json="{}"
            )
        )
    )

    sent_data = await harness.wait_for_response()
    resp = sent_data["toolResponse"]
    self.assertEqual(resp["id"], "call_large")
    self.assertGreaterEqual(len(resp["responseJson"]), 5 * 1024 * 1024)

  async def test_question_hook_integration(self):
    hr = hook_runner.HookRunner()

    @hooks_base.on_interaction
    async def auto_answer(_):
      return hooks_base.QuestionHookResult(
          responses=[
              types.QuestionResponse(selected_option_ids=["1"]),
          ]
      )

    hr.register_hook(auto_answer)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            questions_request=localharness_pb2.UserQuestionsRequest(
                questions=[
                    localharness_pb2.UserQuestion(
                        multiple_choice=localharness_pb2.MultipleChoice(
                            question="Do you agree?",
                            choices=["Yes", "No"],
                        )
                    )
                ]
            ),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("questionResponse", sent_data)
    self.assertEqual(sent_data["questionResponse"]["trajectoryId"], "test_traj")

  async def test_question_hook_integration_unhandled_question(self):
    hr = hook_runner.HookRunner()

    @hooks_base.on_interaction
    async def auto_answer(_):
      return hooks_base.QuestionHookResult(
          responses=[
              types.QuestionResponse(selected_option_ids=["1"]),
          ]
      )

    hr.register_hook(auto_answer)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            questions_request=localharness_pb2.UserQuestionsRequest(
                questions=[
                    localharness_pb2.UserQuestion(
                        multiple_choice=localharness_pb2.MultipleChoice(
                            question="Do you agree?",
                            choices=["Yes", "No"],
                        )
                    ),
                    # Unhandled question type (empty).
                    localharness_pb2.UserQuestion(),
                ]
            ),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("questionResponse", sent_data)
    self.assertEqual(sent_data["questionResponse"]["trajectoryId"], "test_traj")

    resp = sent_data["questionResponse"]["response"]
    self.assertIn("answers", resp)
    self.assertEqual(len(resp["answers"]), 2)

    # First answer should be from hook (selected option 1)
    self.assertIn("multipleChoiceAnswer", resp["answers"][0])
    self.assertEqual(
        resp["answers"][0]["multipleChoiceAnswer"]["selectedChoiceIndices"], [0]
    )

    # Second answer should be unanswered
    self.assertTrue(resp["answers"][1].get("unanswered"))

  async def test_question_hook_integration_empty_questions(self):
    harness = self._make_harness()

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            questions_request=localharness_pb2.UserQuestionsRequest(
                questions=[]
            ),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("questionResponse", sent_data)
    self.assertEqual(sent_data["questionResponse"]["trajectoryId"], "test_traj")

    resp = sent_data["questionResponse"]["response"]
    self.assertEqual(resp, {})

  async def test_yielding_wait_state_to_queue(self):
    """Verifies that wait states are correctly yielded to the step queue for the UI to render."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=5,
            trajectory_id="ui_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            text="Waiting for confirmation",
        )
    )

    await harness.send_event(event)

    # We should be able to retrieve this step from the queue
    step_obj = await asyncio.wait_for(
        harness.conn._step_queue.get(), timeout=2.0
    )
    self.assertEqual(step_obj.trajectory_id, "ui_traj")
    self.assertEqual(step_obj.id, "ui_traj:5")
    self.assertEqual(step_obj.status, types.StepStatus.WAITING_FOR_USER)
    self.assertEqual(step_obj.content, "Waiting for confirmation")

  async def test_cancel_e2e_raises_cancelled_error(self):
    """Verifies programmatic cancel raises AntigravityCancelledError."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

    # Start the turn
    await harness.conn.send("Hello")
    init_data = await harness.wait_for_response()
    self.assertEqual(init_data.get("userInput"), {"parts": [{"text": "Hello"}]})

    # Simulate an active generation step from the harness
    event1 = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="my_cascade",
            trajectory_id="my_cascade",
            step_index=1,
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            text="I'm working",
        )
    )
    await harness.send_event(event1)

    # Consume the steps in a background task to capture yielded output
    steps = []
    receive_error = None

    async def consume() -> None:
      nonlocal receive_error
      try:
        async for step in harness.conn.receive_steps():
          steps.append(step)
      except asyncio.CancelledError as e:
        receive_error = e

    consume_task = asyncio.create_task(consume())

    # Let the background consumer loop spin once
    await asyncio.sleep(0.1)

    # Programmatically cancel the turn
    await harness.conn.cancel()

    # Verify that a halt_request was sent to the backend
    sent_data = await harness.wait_for_response()
    self.assertTrue(sent_data.get("haltRequest"))

    # Simulate the harness transitioning to idle after halting
    event2 = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="my_cascade",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
        )
    )
    await harness.send_event(event2)

    # Await background consumer task completion
    await consume_task

    # Verify yielded step and ensure AntigravityCancelledError propagates
    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "I'm working")
    self.assertIsInstance(receive_error, types.AntigravityCancelledError)

  async def test_handle_tool_call_queues_step(self):
    """Tests ensuring _handle_tool_call manually queues the ToolCall step in _step_queue."""
    harness = self._make_harness()
    conn = harness.conn

    # Mock tool_call protobuf message from WebSocket
    raw_tool_call = localharness_pb2.ToolCall(
        id="call_123",
        name="view_file",
        arguments_json='{"path": "README.md"}',
    )

    # Trigger connection event dispatch
    await conn._handle_tool_call(raw_tool_call)
    await asyncio.sleep(0.1)

    self.assertFalse(conn._step_queue.empty())
    step_obj = await conn._step_queue.get()

    actual_properties = {
        "id": step_obj.id,
        "type": step_obj.type,
        "source": step_obj.source,
        "target": step_obj.target,
        "status": step_obj.status,
        "tool_calls": [
            {"name": tc.name, "args": tc.args} for tc in step_obj.tool_calls
        ],
    }

    expected_properties = {
        "id": "call_123",
        "type": types.StepType.TOOL_CALL,
        "source": types.StepSource.MODEL,
        "target": types.StepTarget.ENVIRONMENT,
        "status": types.StepStatus.ACTIVE,
        "tool_calls": [{"name": "view_file", "args": {"path": "README.md"}}],
    }

    self.assertEqual(actual_properties, expected_properties)

  async def test_wait_for_idle_does_not_deadlock(self):
    """Verifies that wait_for_idle completes when the connection goes idle.

    This test reproduces a bug where wait_for_idle blocks indefinitely if the
    connection becomes idle while receive_steps is awaiting the step queue.
    It also verifies that wait_for_idle supports multiple concurrent callers.
    """
    harness = self._make_harness()
    harness.conn._main_trajectory_id = "parent_traj"
    harness.conn._is_idle.clear()
    harness.conn._parent_idle = False

    # 1. Send an active step update
    await harness.send_event(
        localharness_pb2.OutputEvent(
            step_update=localharness_pb2.StepUpdate(
                cascade_id="parent_traj",
                trajectory_id="parent_traj",
                step_index=1,
                text="Hello",
                state=localharness_pb2.StepUpdate.STATE_ACTIVE,
                source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            )
        )
    )

    # Start multiple wait_for_idle tasks concurrently to verify they all unblock
    wait_task_1 = asyncio.create_task(harness.conn.wait_for_idle())
    wait_task_2 = asyncio.create_task(harness.conn.wait_for_idle())

    # Give tasks time to block
    await asyncio.sleep(0.1)

    # 2. Send trajectory_state_update indicating parent went idle
    await harness.send_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="parent_traj",
                state=localharness_pb2.TrajectoryStateUpdate.State.STATE_FULLY_IDLE,
            )
        )
    )

    # 3. Wait for all wait_tasks to finish with a timeout.
    try:
      await asyncio.wait_for(
          asyncio.gather(wait_task_1, wait_task_2), timeout=2.0
      )
    except asyncio.TimeoutError:
      self.fail("wait_for_idle deadlocked!")

  async def test_concurrent_receive_steps_raises_runtime_error(self):
    """Verifies that concurrent receive_steps() calls raise RuntimeError.

    This test ensures that the SDK prevents multiple consumers from iterating
    over receive_steps() simultaneously. Because steps are drained from a
    single FIFO queue, concurrent iterations would steal steps from one
    another and corrupt conversation history. The active reader guard
    guarantees that a second consumer fails fast with an explicit exception.
    """
    harness = self._make_harness()
    harness.conn._is_idle.clear()

    async def consume_partially() -> None:
      async for _ in harness.conn.receive_steps():
        break

    # Start first consumer in a background task
    consumer_task = asyncio.create_task(consume_partially())
    await asyncio.sleep(0.05)

    # Attempting to start second consumer concurrently raises RuntimeError
    with self.assertRaisesRegex(
        RuntimeError, r"Concurrent receive_steps\(\) calls are not supported"
    ):
      async for _ in harness.conn.receive_steps():
        pass

    # Clean up background task
    await harness.send_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="traj_1",
                state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
            )
        )
    )
    await asyncio.wait_for(consumer_task, timeout=1.0)

  async def test_last_turn_stop_reason(self):
    harness = self._make_harness()
    self.assertEqual(
        harness.conn._last_turn_stop_reason,
        types.StopReason.UNSPECIFIED,
    )
    harness.conn._processor._turn_stop_reason = (
        types.StopReason.QUOTA_EXHAUSTED
    )
    self.assertEqual(
        harness.conn._last_turn_stop_reason,
        types.StopReason.QUOTA_EXHAUSTED,
    )


class LocalConnectionToolCallNoRunnerTest(unittest.IsolatedAsyncioTestCase):
  """Tests for tool call handling when no ToolRunner is configured."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()

  async def test_tool_call_without_runner_yields_step(self):
    """Verifies that a tool call with no ToolRunner queues a step for the user.

    Why: When no ToolRunner is configured, the connection should surface the
    tool call to the caller so they can handle it manually, rather than
    silently dropping it.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=None,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_99",
            name="custom_tool",
            arguments_json='{"key": "value"}',
        )
    )

    await harness.send_event(event)

    step_obj = await asyncio.wait_for(
        harness.conn._step_queue.get(), timeout=1.0
    )
    self.assertEqual(step_obj.type, types.StepType.TOOL_CALL)
    self.assertEqual(step_obj.tool_calls[0].name, "custom_tool")
    self.assertEqual(step_obj.tool_calls[0].args, {"key": "value"})
    self.assertEqual(step_obj.tool_calls[0].id, "call_99")
    # No messages should have been sent back to the harness.
    self.assertEqual(len(harness.ws.sent_messages), 0)


class LocalConnectionStrategyConfigTest(parameterized.TestCase):
  """Tests for config-to-proto translation in LocalConnectionStrategy.

  These tests exercise _build_harness_config() directly, without mocking
  any internal logic. Only the strategy constructor and config builder run;
  no subprocess or websocket I/O is triggered.
  """

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)

  def _make_strategy(self, **kwargs):
    """Creates a LocalConnectionStrategy with the given kwargs."""
    return local_connection.LocalConnectionStrategy(**kwargs)

  def test_default_config_produces_valid_harness_config(self):
    """Verifies that a strategy with all defaults produces a well-formed proto.

    Why: The default path is the most common case. Callers should be able to
    construct a strategy with only binary_path and get a valid HarnessConfig.
    How: Build the config and assert the proto has expected default structure.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertIsInstance(config, localharness_pb2.HarnessConfig)
    # Default: all harness side tools enabled.
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    self.assertTrue(config.harness_side_tools.user_questions.enabled)
    self.assertTrue(config.harness_side_tools.run_command.enabled)
    self.assertTrue(config.harness_side_tools.find.enabled)
    self.assertTrue(config.harness_side_tools.generate_image.enabled)
    # No models, system instructions, workspaces, or skills by default.
    self.assertEmpty(config.models)
    self.assertFalse(config.HasField("system_instructions"))
    self.assertEmpty(config.workspaces)
    self.assertEqual(
        config.agent_behavior, localharness_pb2.AGENT_BEHAVIOR_AUTONOMOUS
    )

  def test_agent_behavior_config_produces_valid_proto(self):
    """Verifies that agent_behavior sets HarnessConfig.agent_behavior."""
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            agent_behavior=types.AgentBehavior.INTERACTIVE
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.agent_behavior, localharness_pb2.AGENT_BEHAVIOR_INTERACTIVE
    )

  def test_subagent_agent_behavior_config_produces_valid_proto(self):
    """Verifies that SubagentCapabilities.agent_behavior sets CustomAgent.agent_behavior."""
    strategy = self._make_strategy(
        subagents=[
            types.SubagentConfig(
                name="interactive_subagent",
                description="A subagent that runs in interactive behavior.",
                model="gemini-2.5-pro",
                capabilities=types.SubagentCapabilities(
                    agent_behavior=types.AgentBehavior.INTERACTIVE
                ),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertLen(config.custom_subagents, 1)
    self.assertEqual(
        config.custom_subagents[0].agent_behavior,
        localharness_pb2.AGENT_BEHAVIOR_INTERACTIVE,
    )

  def test_subagent_minimal_agent_behavior_config_produces_valid_proto(self):
    """Verifies that SubagentCapabilities.agent_behavior=MINIMAL sets CustomAgent.agent_behavior."""
    strategy = self._make_strategy(
        subagents=[
            types.SubagentConfig(
                name="minimal_subagent",
                description="A subagent that runs in minimal behavior.",
                model="gemini-3.8-flash",
                capabilities=types.SubagentCapabilities(
                    agent_behavior=types.AgentBehavior.MINIMAL
                ),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertLen(config.custom_subagents, 1)
    self.assertEqual(
        config.custom_subagents[0].agent_behavior,
        localharness_pb2.AGENT_BEHAVIOR_MINIMAL,
    )

  def test_legacy_shorthands_api_key_produces_valid_proto(self):
    """Verifies that the legacy api_key shorthand translates to the models proto."""
    cfg = local_connection_config.LocalAgentConfig(
        model="gemini-2.5-flash",
        api_key="shorthand-key",
    )
    strategy = self._make_strategy(
        models=cfg.models,
    )
    config = strategy._build_harness_config()
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "gemini-2.5-flash")
    self.assertTrue(config.models[0].HasField("gemini_api_endpoint"))
    self.assertEqual(
        config.models[0].gemini_api_endpoint.api_key, "shorthand-key"
    )
    self.assertEqual(config.models[0].types, [localharness_pb2.MODEL_TYPE_TEXT])

  def test_legacy_shorthands_vertex_produces_valid_proto(self):
    """Verifies that the legacy vertex shorthands translate to the models proto."""
    cfg = local_connection_config.LocalAgentConfig(
        model="gemini-2.5-flash",
        vertex=True,
        project="vertex-project",
        location="us-east4",
    )
    strategy = self._make_strategy(
        models=cfg.models,
    )
    config = strategy._build_harness_config()
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "gemini-2.5-flash")
    self.assertTrue(config.models[0].HasField("vertex_endpoint"))
    self.assertEqual(config.models[0].vertex_endpoint.project, "vertex-project")
    self.assertEqual(config.models[0].vertex_endpoint.location, "us-east4")

  def test_capabilities_config_finish_tool_schema_json_to_proto(self):
    """Verifies capabilities config propagates finish tool schema to the proto config.

    Why: The user's custom schema must be delivered to the Go harness so it can
    be appropriately injected into the finish tool declaration.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            finish_tool_schema_json='{"type": "object"}',
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.finish_tool_schema_json, '{"type": "object"}')

  def test_gemini_config_to_proto(self):
    """Verifies GeminiConfig fields translate to the correct proto fields.

    Why: The proto's field names must match the Pydantic model's semantics
    exactly, or the Go harness will receive incorrect configuration.
    How: Set all GeminiConfig fields and assert proto field values.
    """
    strategy = self._make_strategy(
        models=[
            types.ModelTarget(
                name="gemini-2.5-pro",
                types=[types.ModelType.TEXT],
                endpoint=types.GeminiAPIEndpoint(api_key="test-key"),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.models[0].gemini_api_endpoint.api_key, "test-key")
    self.assertEqual(config.models[0].name, "gemini-2.5-pro")

  def test_gemini_config_none_fields_omitted(self):
    """Verifies that None fields on ModelConfig are not set on the proto."""
    models = [
        types.ModelTarget(
            name="gemini-3.8-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(),
        )
    ]
    strategy = self._make_strategy(models=models)
    config = strategy._build_harness_config()
    self.assertEqual(config.models[0].name, "gemini-3.8-flash")
    # api_key should not be set (proto default empty string).
    self.assertEqual(config.models[0].gemini_api_endpoint.api_key, "")

  def test_models_default_model_name(self):
    """Verifies the default model name propagates correctly."""
    models = [
        types.ModelTarget(
            name=None,
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(),
        )
    ]
    strategy = self._make_strategy(models=models)
    config = strategy._build_harness_config()
    self.assertEqual(config.models[0].name, "")

  def test_system_instructions_string_shorthand(self):
    """Verifies that a plain string normalizes to AppendedSystemInstructions.

    Why: The str shorthand is an ergonomic convenience. It defaults to
    appending.
    How: Pass a string, build proto, and assert the appended field is set.
    """
    strategy = self._make_strategy(system_instructions="Be concise.")
    config = strategy._build_harness_config()
    self.assertLen(
        config.system_instructions.appended.appended_sections, 1
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].content,
        "Be concise.",
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].title,
        "user_system_instructions",
    )

  def test_system_instructions_model_custom(self):
    """Verifies that CustomSystemInstructions sets custom on the proto."""
    strategy = self._make_strategy(
        system_instructions=types.CustomSystemInstructions(
            text="Override everything."
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.system_instructions.custom.part[0].text, "Override everything."
    )

  def test_system_instructions_model_templated(self):
    """Verifies that TemplatedSystemInstructions sets appended on the proto."""
    section = types.SystemInstructionSection(
        title="extra", content="More instructions"
    )
    strategy = self._make_strategy(
        system_instructions=types.TemplatedSystemInstructions(
            identity="New Identity", sections=[section]
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.system_instructions.appended.custom_identity, "New Identity"
    )
    self.assertLen(
        config.system_instructions.appended.appended_sections, 1
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].title, "extra"
    )

  def test_system_instructions_model_templated_only_identity(self):
    """Verifies that TemplatedSystemInstructions with only identity maps correctly."""
    strategy = self._make_strategy(
        system_instructions=types.TemplatedSystemInstructions(
            identity="Only Identity"
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.system_instructions.appended.custom_identity, "Only Identity"
    )
    self.assertEmpty(
        config.system_instructions.appended.appended_sections
    )

  def test_system_instructions_model_templated_only_sections(self):
    """Verifies that TemplatedSystemInstructions with only sections maps correctly."""
    section = types.SystemInstructionSection(
        title="extra", content="More instructions"
    )
    strategy = self._make_strategy(
        system_instructions=types.TemplatedSystemInstructions(
            sections=[section]
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.system_instructions.appended.custom_identity, "")
    self.assertLen(
        config.system_instructions.appended.appended_sections, 1
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].title, "extra"
    )

  def test_system_instructions_none(self):
    """Verifies that no system_instructions field is set when not provided.

    Why: The harness should use its own defaults when no instructions are given.
    How: Build with system_instructions=None and assert no proto field is set.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertFalse(config.HasField("system_instructions"))

  def test_to_system_instructions_proto_direct_string(self):
    """Verifies that to_system_instructions_proto directly normalizes str to AppendedSystemInstructions."""
    proto = local_connection.to_system_instructions_proto(
        "Direct string instructions"
    )
    self.assertIsNotNone(proto)
    self.assertTrue(proto.HasField("appended"))
    self.assertLen(proto.appended.appended_sections, 1)
    self.assertEqual(
        proto.appended.appended_sections[0].content,
        "Direct string instructions",
    )

  def test_workspaces_to_proto(self):
    """Verifies workspace paths translate to Workspace protos correctly.

    Why: The harness uses a structured Workspace proto with FilesystemWorkspace;
    plain strings must be wrapped correctly.
    How: Pass two paths via session_config, build proto, and assert each
    workspace directory.
    """
    strategy = self._make_strategy(
        workspaces=["/home/user/project", "/tmp/scratch"]
    )
    config = strategy._build_harness_config()
    self.assertLen(config.workspaces, 2)
    self.assertEqual(
        config.workspaces[0].filesystem_workspace.directory,
        "/home/user/project",
    )
    self.assertEqual(
        config.workspaces[1].filesystem_workspace.directory,
        "/tmp/scratch",
    )

  def test_workspaces_none(self):
    """Verifies that no workspaces are set when not provided.

    Why: The harness should not receive spurious workspace entries.
    How: Build with default session_config and assert empty repeated field.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertEmpty(config.workspaces)

  def test_empty_workspaces_list(self):
    """Verifies that an empty list produces an empty repeated field.

    Why: workspaces=[] is a valid explicit choice meaning 'no workspaces',
    distinct from None (which also means no workspaces but is implicit).
    How: Pass empty list via session_config and assert empty repeated field.
    """
    strategy = self._make_strategy(workspaces=[])
    config = strategy._build_harness_config()
    self.assertEmpty(config.workspaces)

  def test_skills_paths_to_proto(self):
    """Verifies skills_paths translate directly to the proto repeated field.

    Why: Skills paths are simple strings that map 1:1 to the proto field.
    How: Pass a list and assert proto field contents.
    """
    strategy = self._make_strategy(skills_paths=["/skills/a", "/skills/b"])
    config = strategy._build_harness_config()
    self.assertEqual(list(config.skills_paths), ["/skills/a", "/skills/b"])

  def test_capabilities_config_disabled_tools(self):
    """Verifies that disabling tools produces the correct proto.

    Why: Each BuiltinTool with a proto toggle should map to its config field.
    How: Disable RUN_COMMAND and ASK_QUESTION and assert each sub-proto's
    enabled field, plus check that other tools remain enabled.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            disabled_tools=[
                types.BuiltinTools.RUN_COMMAND,
                types.BuiltinTools.ASK_QUESTION,
                types.BuiltinTools.GENERATE_IMAGE,
            ],
        )
    )
    config = strategy._build_harness_config()
    self.assertFalse(config.harness_side_tools.run_command.enabled)
    self.assertFalse(config.harness_side_tools.user_questions.enabled)
    self.assertFalse(config.harness_side_tools.generate_image.enabled)
    # Subagents were not disabled; should still be enabled by default.
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    # Tools that were not disabled should still be enabled.
    self.assertTrue(config.harness_side_tools.find.enabled)
    self.assertTrue(config.harness_side_tools.file_edit.enabled)
    self.assertTrue(config.harness_side_tools.view_file.enabled)
    self.assertTrue(config.harness_side_tools.write_to_file.enabled)
    self.assertTrue(config.harness_side_tools.grep_search.enabled)
    self.assertTrue(config.harness_side_tools.list_dir.enabled)
    self.assertTrue(config.harness_side_tools.search_web.enabled)
    self.assertTrue(config.harness_side_tools.read_url_content.enabled)

  def test_capabilities_config_enabled_tools(self):
    """Verifies that enabled_tools allowlist excludes non-listed tools.

    Why: When an explicit allowlist is provided, only those tools should be
    active; all others should be disabled at the proto level.
    How: Enable only VIEW_FILE and assert all other tools are disabled.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            enabled_tools=[types.BuiltinTools.VIEW_FILE],
        )
    )
    config = strategy._build_harness_config()

    expected_harness_side_tools = localharness_pb2.HarnessSideTools(
        view_file=localharness_pb2.ViewFileToolConfig(enabled=True),
        subagents=localharness_pb2.SubagentsConfig(enabled=False),
        user_questions=localharness_pb2.UserQuestionsConfig(enabled=False),
        run_command=localharness_pb2.RunCommandToolConfig(
            enabled=False,
            enable_daemon_commands=False,
            max_timeout_ms=0,
            enable_sandbox=False,
        ),
        find=localharness_pb2.FindToolConfig(enabled=False),
        generate_image=localharness_pb2.GenerateImageToolConfig(enabled=False),
        file_edit=localharness_pb2.FileEditToolConfig(enabled=False),
        write_to_file=localharness_pb2.WriteToFileToolConfig(enabled=False),
        grep_search=localharness_pb2.GrepSearchToolConfig(enabled=False),
        list_dir=localharness_pb2.ListDirToolConfig(enabled=False),
        search_web=localharness_pb2.SearchWebToolConfig(enabled=False),
        read_url_content=localharness_pb2.ReadUrlContentToolConfig(
            enabled=False
        ),
    )

    self.assertEqual(config.harness_side_tools, expected_harness_side_tools)

  def test_capabilities_config_run_command_custom(self):
    """Verifies RunCommandConfig maps to RunCommandToolConfig."""
    strategy_custom = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            run_command_config=types.RunCommandConfig(
                enable_daemons=False,
                timeout_seconds=60.0,
            ),
        )
    )
    config_custom = strategy_custom._build_harness_config()
    self.assertFalse(
        config_custom.harness_side_tools.run_command.enable_daemon_commands
    )
    self.assertEqual(
        config_custom.harness_side_tools.run_command.max_timeout_ms,
        60_000,
    )

    strategy_true = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            run_command_config=types.RunCommandConfig(
                enable_daemons=True,
                timeout_seconds=None,
            ),
        )
    )
    config_true = strategy_true._build_harness_config()
    self.assertTrue(
        config_true.harness_side_tools.run_command.enable_daemon_commands
    )
    self.assertEqual(
        config_true.harness_side_tools.run_command.max_timeout_ms,
        0,
    )

  def test_build_harness_config_defaults_run_command(self):
    """Verifies run_command defaults when capabilities_config is None or has no RunCommandConfig."""
    strategy = self._make_strategy(capabilities_config=None)
    config = strategy._build_harness_config()
    self.assertFalse(
        config.harness_side_tools.run_command.enable_daemon_commands
    )
    self.assertEqual(
        config.harness_side_tools.run_command.max_timeout_ms,
        0,
    )

  def test_capabilities_config_run_command_sandbox_forwarded(self):
    """Verifies the OS-sandbox opt-in maps onto RunCommandToolConfig.

    Why: The ACP server sets run_command_config.enable_sandbox to honor the
    enterprise PROCEED_IN_SANDBOX / sandbox_mode admin controls; it must reach
    the harness/cortex over the wire.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            run_command_config=types.RunCommandConfig(enable_sandbox=True),
        )
    )
    config = strategy._build_harness_config()

    run_command = config.harness_side_tools.run_command
    self.assertTrue(run_command.enabled)
    self.assertTrue(run_command.enable_sandbox)

  def test_capabilities_config_run_command_sandbox_defaults_off(self):
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig()
    )
    config = strategy._build_harness_config()

    run_command = config.harness_side_tools.run_command
    self.assertFalse(run_command.enable_sandbox)

  def test_capabilities_config_max_subagent_depth_and_allowed_subagents(self):
    """Verifies max_subagent_depth and allowed_subagents map to SubagentsConfig."""
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            max_subagent_depth=3,
            allowed_subagents=["researcher", "reviewer"],
        )
    )
    config = strategy._build_harness_config()
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    self.assertEqual(config.harness_side_tools.subagents.max_nesting_depth, 3)
    self.assertEqual(
        list(config.harness_side_tools.subagents.allowed_subagents),
        ["researcher", "reviewer"],
    )

  def test_build_custom_subagents_protos_nested_subagent_enabled(self):
    """Verifies custom subagents can have subagents enabled and scoped."""
    child_subagent = types.SubagentConfig(
        name="researcher",
        description="Does research",
        capabilities=types.SubagentCapabilities(
            enabled_tools=[types.BuiltinTools.START_SUBAGENT],
            allowed_subagents=["fact_checker"],
        ),
    )
    strategy = self._make_strategy(subagents=[child_subagent])
    custom_agents = strategy._build_custom_subagents_protos({})
    self.assertLen(custom_agents, 1)
    self.assertEqual(custom_agents[0].name, "researcher")
    self.assertTrue(custom_agents[0].harness_side_tools.subagents.enabled)
    self.assertEqual(
        list(custom_agents[0].harness_side_tools.subagents.allowed_subagents),
        ["fact_checker"],
    )

  def test_capabilities_config_compaction_threshold(self):
    """Verifies compaction_threshold maps to HarnessConfig.compaction_threshold.

    Why: This controls context window compaction behavior in the harness.
    How: Set a threshold and assert it appears on the proto.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(compaction_threshold=50000)
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.compaction_threshold, 50000)
    self.assertTrue(config.HasField("compaction_config"))
    self.assertEqual(config.compaction_config.checkpoint_interval_tokens, 50000)

  def test_capabilities_config_none_uses_defaults(self):
    """Verifies that capabilities_config=None produces default-enabled tools.

    Why: The most common case is no explicit CapabilitiesConfig; all tools
    should be enabled and compaction_threshold unset.
    How: Build with no capabilities_config and assert defaults.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    self.assertTrue(config.harness_side_tools.user_questions.enabled)
    self.assertTrue(config.harness_side_tools.run_command.enabled)
    self.assertTrue(config.harness_side_tools.find.enabled)
    self.assertEqual(config.compaction_threshold, 0)
    self.assertFalse(config.HasField("compaction_config"))

  def test_compaction_config_explicit(self):
    """Verifies CompactionConfig maps to HarnessConfig.compaction_config."""
    strategy = self._make_strategy(
        compaction_config=types.CompactionConfig(
            checkpoint_interval_tokens=40000, max_context_tokens=80000
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.compaction_threshold, 40000)
    self.assertTrue(config.HasField("compaction_config"))
    self.assertEqual(config.compaction_config.checkpoint_interval_tokens, 40000)
    self.assertEqual(config.compaction_config.max_context_tokens, 80000)

  def test_compaction_config_precedence_over_capabilities(self):
    """Verifies that compaction_config takes precedence over CapabilitiesConfig."""
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            compaction_threshold=50000
        ),
        compaction_config=types.CompactionConfig(
            checkpoint_interval_tokens=30000, max_context_tokens=60000
        ),
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.compaction_threshold, 30000)
    self.assertTrue(config.HasField("compaction_config"))
    self.assertEqual(config.compaction_config.checkpoint_interval_tokens, 30000)
    self.assertEqual(config.compaction_config.max_context_tokens, 60000)

  def test_cascade_id_passed_through(self):
    """Verifies that session_config.conversation_id maps to HarnessConfig.cascade_id.

    Why: cascade_id is used for session resumption; if it's lost, the
    harness creates a new session instead of resuming.
    How: Set conversation_id via session_config and assert it appears
    on the proto.
    """
    strategy = self._make_strategy(
        conversation_id="12345678901234567890123456789012"
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.cascade_id, "12345678901234567890123456789012")

  def test_session_continuation_mode_passed_through(self):
    """Verifies session_continuation_mode maps to proto."""
    for sdk_mode, proto_mode in [
        (
            types.SessionContinuationMode.RESUME,
            localharness_pb2.HarnessConfig.RESUME,
        ),
        (
            types.SessionContinuationMode.CREATE_OR_RESUME,
            localharness_pb2.HarnessConfig.CREATE_OR_RESUME,
        ),
        (
            types.SessionContinuationMode.CREATE_ONLY,
            localharness_pb2.HarnessConfig.CREATE_ONLY,
        ),
    ]:
      with self.subTest(sdk_mode=sdk_mode):
        strategy = self._make_strategy(session_continuation_mode=sdk_mode)
        config = strategy._build_harness_config()
        self.assertEqual(config.session_continuation_mode, proto_mode)

  def test_session_continuation_mode_default_unspecified(self):
    """Verifies session_continuation_mode defaults to UNSPECIFIED.

    Why: If session_continuation_mode is not explicitly set, the harness should
      use its default fallback logic.
    How: Build with default config and assert session_continuation_mode is
      UNSPECIFIED.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertEqual(
        config.session_continuation_mode,
        localharness_pb2.HarnessConfig.SESSION_CONTINUATION_MODE_UNSPECIFIED,
    )

  def test_cascade_id_default_empty(self):
    """Verifies that cascade_id defaults to empty string when no conversation_id set.

    Why: The harness treats an empty cascade_id as a fresh session.
    How: Build with default session_config and assert empty cascade_id.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertEqual(config.cascade_id, "")

  def test_storage_directory_from_save_dir(self):
    """Verifies save_dir maps to InputConfig.storage_directory.

    Why: The harness writes trajectory data to storage_directory. If
    save_dir is silently dropped, session state is never persisted and
    resumption breaks.
    How: Set save_dir via session_config and assert it appears on
    the strategy's stored config for InputConfig construction.
    """
    strategy = self._make_strategy(save_dir="/tmp/state")
    self.assertEqual(strategy._save_dir, "/tmp/state")

  def test_storage_directory_defaults_to_none(self):
    """Verifies save_dir is None when not specified.

    Why: A None save_dir signals an ephemeral session. The or "" fallback
    in __aenter__ must produce an empty string for the proto.
    How: Build with default session_config and assert save_dir is None.
    """
    strategy = self._make_strategy()
    self.assertIsNone(strategy._save_dir)

  def test_workspaces_default_empty(self):
    """Verifies no workspace protos when session_config has no workspaces.

    Why: The or [] fallback prevents iterating over None. If removed,
    the list comprehension raises TypeError on None.
    How: Build with default session_config and assert empty workspaces.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertEmpty(config.workspaces)

  def test_models_thinking_level_set(self):
    """Verifies that thinking_level on ModelTarget maps to the proto field."""
    strategy = self._make_strategy(
        models=[
            types.ModelTarget(
                name=models_lib.DEFAULT_MODEL,
                types=[types.ModelType.TEXT],
                endpoint=types.GeminiAPIEndpoint(
                    options=types.GeminiModelOptions(
                        thinking_level=types.ThinkingLevel.HIGH,
                    ),
                ),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.models[0].gemini_api_endpoint.options.thinking_level, "high"
    )

  def test_models_thinking_level_none_omitted(self):
    """Verifies that thinking_level=None leaves the proto field at its default."""
    strategy = self._make_strategy(
        models=[
            types.ModelTarget(
                name=models_lib.DEFAULT_MODEL,
                types=[types.ModelType.TEXT],
                endpoint=types.GeminiAPIEndpoint(
                    options=types.GeminiModelOptions(thinking_level=None),
                ),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertFalse(config.models[0].gemini_api_endpoint.HasField("options"))

  def test_models_thinking_level_all_values(self):
    """Verifies all ThinkingLevel enum values produce correct proto strings."""
    for level in types.ThinkingLevel:
      strategy = self._make_strategy(
          models=[
              types.ModelTarget(
                  name=models_lib.DEFAULT_MODEL,
                  types=[types.ModelType.TEXT],
                  endpoint=types.GeminiAPIEndpoint(
                      options=types.GeminiModelOptions(thinking_level=level),
                  ),
              )
          ]
      )
      config = strategy._build_harness_config()
      self.assertEqual(
          config.models[0].gemini_api_endpoint.options.thinking_level,
          level.value,
          f"ThinkingLevel.{level.name} should produce proto string"
          f" '{level.value}'",
      )

  def test_models_service_tier_set(self):
    """Verifies that service_tier on ModelTarget maps to the proto field."""
    strategy = self._make_strategy(
        models=[
            types.ModelTarget(
                name=models_lib.DEFAULT_MODEL,
                types=[types.ModelType.TEXT],
                endpoint=types.GeminiAPIEndpoint(
                    options=types.GeminiModelOptions(
                        service_tier=types.ServiceTier.PRIORITY,
                    ),
                ),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.models[0].gemini_api_endpoint.options.service_tier, "priority"
    )

  def test_models_service_tier_none_omitted(self):
    """Verifies that service_tier=None leaves the proto field at its default."""
    strategy = self._make_strategy(
        models=[
            types.ModelTarget(
                name=models_lib.DEFAULT_MODEL,
                types=[types.ModelType.TEXT],
                endpoint=types.GeminiAPIEndpoint(
                    options=types.GeminiModelOptions(service_tier=None),
                ),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertFalse(config.models[0].gemini_api_endpoint.HasField("options"))

  def test_models_service_tier_all_values(self):
    """Verifies all ServiceTier enum values produce correct proto strings."""
    for tier in types.ServiceTier:
      strategy = self._make_strategy(
          models=[
              types.ModelTarget(
                  name=models_lib.DEFAULT_MODEL,
                  types=[types.ModelType.TEXT],
                  endpoint=types.GeminiAPIEndpoint(
                      options=types.GeminiModelOptions(service_tier=tier),
                  ),
              )
          ]
      )
      config = strategy._build_harness_config()
      self.assertEqual(
          config.models[0].gemini_api_endpoint.options.service_tier,
          tier.value,
          f"ServiceTier.{tier.name} should produce proto string '{tier.value}'",
      )

  def test_vertex_endpoint_service_tier_set(self):
    """Verifies that service_tier on VertexEndpoint maps to the proto field."""
    strategy = self._make_strategy(
        models=[
            types.ModelTarget(
                name=models_lib.DEFAULT_MODEL,
                types=[types.ModelType.TEXT],
                endpoint=types.VertexEndpoint(
                    project="test-project",
                    location="us-central1",
                    options=types.GeminiModelOptions(
                        service_tier=types.ServiceTier.PRIORITY,
                    ),
                ),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.models[0].vertex_endpoint.options.service_tier, "priority"
    )

  def test_vertex_config_propagates(self):
    """Verifies that Vertex configuration fields propagate to proto."""
    models = [
        types.ModelTarget(
            name="gemini-3.8-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.VertexEndpoint(
                project="my-project",
                location="us-central1",
            ),
        )
    ]
    strategy = self._make_strategy(models=models)
    config = strategy._build_harness_config()
    self.assertTrue(config.models[0].HasField("vertex_endpoint"))
    self.assertEqual(config.models[0].vertex_endpoint.project, "my-project")
    self.assertEqual(config.models[0].vertex_endpoint.location, "us-central1")

  def test_models_list_propagates(self):
    """Verifies that the new models list propagates to HarnessConfig."""
    models = [
        types.ModelTarget(
            name="gemini-2.5-pro",
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(
                options=types.GeminiModelOptions(
                    thinking_level=types.ThinkingLevel.HIGH
                )
            ),
        ),
        types.ModelTarget(
            name="imagen-3-custom",
            types=[types.ModelType.IMAGE],
            endpoint=types.GeminiAPIEndpoint(),
        ),
    ]
    strategy = self._make_strategy(models=models)
    config = strategy._build_harness_config()

    # Text model assertions
    self.assertEqual(config.models[0].name, "gemini-2.5-pro")
    self.assertEqual(
        config.models[0].gemini_api_endpoint.options.thinking_level, "high"
    )

    # Image model assertions
    self.assertEqual(config.models[1].name, "imagen-3-custom")
    self.assertEqual(
        config.models[1].types, [localharness_pb2.MODEL_TYPE_IMAGE]
    )

  def test_models_list_custom_endpoints_propagate(self):
    """Verifies that custom endpoints in the models list propagate to proto."""
    # Test VertexEndpoint
    vertex_endpoint = types.VertexEndpoint(
        project="vertex-proj", location="europe-west1"
    )
    models_vertex = [
        types.ModelTarget(
            name="gemini-ultra",
            types=[types.ModelType.TEXT],
            endpoint=vertex_endpoint,
        )
    ]
    strategy = self._make_strategy(models=models_vertex)
    config = strategy._build_harness_config()
    self.assertTrue(config.models[0].HasField("vertex_endpoint"))
    self.assertEqual(config.models[0].vertex_endpoint.project, "vertex-proj")
    self.assertEqual(config.models[0].vertex_endpoint.location, "europe-west1")

    # Test GeminiAPIEndpoint
    api_endpoint = types.GeminiAPIEndpoint(api_key="api-key-xyz")
    models_api = [
        types.ModelTarget(
            name="gemini-flash",
            types=[types.ModelType.TEXT],
            endpoint=api_endpoint,
        )
    ]
    strategy = self._make_strategy(models=models_api)
    config = strategy._build_harness_config()
    self.assertTrue(config.models[0].HasField("gemini_api_endpoint"))
    self.assertEqual(
        config.models[0].gemini_api_endpoint.api_key, "api-key-xyz"
    )

  def test_models_stored_directly_on_strategy(self):
    """Verifies that the models list is stored as self._models."""
    models = [
        types.ModelTarget(
            name="gemini-2.5-pro",
            types=[types.ModelType.TEXT],
        ),
    ]
    strategy = self._make_strategy(models=models)
    self.assertIsNotNone(strategy._models)
    self.assertLen(strategy._models, 1)
    self.assertEqual(strategy._models[0].name, "gemini-2.5-pro")

  def test_session_config_save_dir_stored(self):
    """Verifies that session_config.save_dir is preserved on the strategy.

    Why: save_dir maps to InputConfig.storage_directory during __aenter__.
    The strategy must store it so the startup sequence can use it.
    How: Set save_dir via session_config and assert strategy attribute.
    """
    strategy = self._make_strategy(save_dir="/data/sessions")
    self.assertEqual(strategy._save_dir, "/data/sessions")

  def test_session_config_save_dir_default_none(self):
    """Verifies that save_dir defaults to None when not provided.

    Why: When no save_dir is set, InputConfig.storage_directory should be
    empty and persistence is disabled.
    How: Build with default session_config and assert save_dir is None.
    """
    strategy = self._make_strategy()
    self.assertIsNone(strategy._save_dir)

  def test_full_session_config_to_proto(self):
    """Verifies that a full session_config produces correct proto fields.

    Why: This is the canonical resumption case — all three session fields
    must map correctly to their proto counterparts.
    How: Set all session_config fields, build proto, and assert each mapping.
    """
    strategy = self._make_strategy(
        conversation_id="session-789",
        save_dir="/state/dir",
        workspaces=["/ws/a"],
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.cascade_id, "session-789")
    self.assertLen(config.workspaces, 1)
    self.assertEqual(
        config.workspaces[0].filesystem_workspace.directory, "/ws/a"
    )
    # save_dir is wired in __aenter__, not _build_harness_config;
    # verify storage.
    self.assertEqual(strategy._save_dir, "/state/dir")

  def test_app_data_dir_specified(self):
    strategy = self._make_strategy(app_data_dir="/custom/app/data")
    config = strategy._build_harness_config()
    self.assertEqual(config.app_data_dir, "/custom/app/data")

  def test_app_data_dir_default_empty(self):
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    # When app_data_dir is not explicitly set, the strategy resolves the
    # default (~/.gemini/antigravity) so Go's PolicyEvaluator can allowlist it.
    expected = str(pathlib.Path("~/.gemini/antigravity").expanduser().resolve())
    self.assertEqual(config.app_data_dir, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="via_disabled_tools",
          capabilities_config=types.CapabilitiesConfig(
              enable_subagents=True,
              disabled_tools=[types.BuiltinTools.START_SUBAGENT],
          ),
      ),
      dict(
          testcase_name="via_enable_subagents_false",
          capabilities_config=types.CapabilitiesConfig(
              enable_subagents=False,
          ),
      ),
  )
  def test_capabilities_config_subagents_disabled(self, capabilities_config):
    """Verifies that subagents are disabled based on capabilities_config."""
    strategy = self._make_strategy(capabilities_config=capabilities_config)
    config = strategy._build_harness_config()
    self.assertFalse(config.harness_side_tools.subagents.enabled)

  def test_strategy_normalizes_configured_workspaces(self):
    """Verifies that workspace configurations using file:// URIs are canonicalized."""
    strategy = self._make_strategy(
        workspaces=["file:///dev/shm/workspace", "/tmp/clean-path"]
    )
    self.assertEqual(
        strategy._workspaces,
        ["/dev/shm/workspace", str(pathlib.Path("/tmp/clean-path").resolve())],
    )

  def test_strategy_normalizes_relative_workspaces(self):
    """Verifies that relative workspace paths and user home ~ are resolved."""
    strategy = self._make_strategy(
        workspaces=["ws", "./subdir", "../parent", "~/my_project"]
    )
    expected = [
        str(pathlib.Path("ws").resolve()),
        str(pathlib.Path("./subdir").resolve()),
        str(pathlib.Path("../parent").resolve()),
        str(pathlib.Path("~/my_project").expanduser().resolve()),
    ]
    self.assertEqual(strategy._workspaces, expected)
    config = strategy._build_harness_config()
    self.assertLen(config.workspaces, 4)
    for i, exp in enumerate(expected):
      self.assertEqual(config.workspaces[i].filesystem_workspace.directory, exp)

  def test_strategy_normalizes_pathlib_paths(self):
    """Verifies that pathlib.Path objects are resolved and accepted."""
    strategy = self._make_strategy(
        workspaces=[pathlib.Path("ws"), pathlib.Path("/dev/shm/ws")]
    )
    expected = [
        str(pathlib.Path("ws").resolve()),
        str(pathlib.Path("/dev/shm/ws").resolve()),
    ]
    self.assertEqual(strategy._workspaces, expected)

  def test_strategy_normalizes_cns_workspaces(self):
    """Verifies that CNS URIs and paths are preserved without filesystem resolve."""
    strategy = self._make_strategy(
        workspaces=["cns://el-d/home/user/project", "/cns/el-d/home/user/data"]
    )
    expected = [
        "/cns/el-d/home/user/project",
        "/cns/el-d/home/user/data",
    ]
    self.assertEqual(strategy._workspaces, expected)
    config = strategy._build_harness_config()
    self.assertEqual(
        config.workspaces[0].filesystem_workspace.directory,
        "/cns/el-d/home/user/project",
    )
    self.assertEqual(
        config.workspaces[1].filesystem_workspace.directory,
        "/cns/el-d/home/user/data",
    )

  def test_local_agent_config_workspaces_validation(self):
    """Verifies LocalAgentConfig validation, coercion, and normalization of workspaces."""
    # Default is [os.getcwd()]
    cfg_default = local_connection_config.LocalAgentConfig()
    self.assertEqual(cfg_default.workspaces, [os.getcwd()])

    # Explicit empty list is preserved
    cfg_empty = local_connection_config.LocalAgentConfig(workspaces=[])
    self.assertEqual(cfg_empty.workspaces, [])

    # Single string is coerced and normalized
    cfg_str = local_connection_config.LocalAgentConfig(workspaces="ws")
    self.assertEqual(cfg_str.workspaces, [str(pathlib.Path("ws").resolve())])

    # Single Path is coerced and normalized
    cfg_path = local_connection_config.LocalAgentConfig(
        workspaces=pathlib.Path("ws")
    )
    self.assertEqual(cfg_path.workspaces, [str(pathlib.Path("ws").resolve())])

    # List of relative paths and ~
    cfg_list = local_connection_config.LocalAgentConfig(
        workspaces=["ws", "./nested", "~/my_project"]
    )
    self.assertEqual(
        cfg_list.workspaces,
        [
            str(pathlib.Path("ws").resolve()),
            str(pathlib.Path("./nested").resolve()),
            str(pathlib.Path("~/my_project").expanduser().resolve()),
        ],
    )

    # URIs
    cfg_uris = local_connection_config.LocalAgentConfig(
        workspaces=["file:///dev/shm/ws", "cns://el-d/home/user/ws"]
    )
    self.assertEqual(
        cfg_uris.workspaces,
        ["/dev/shm/ws", "/cns/el-d/home/user/ws"],
    )

    # Invalid types raise ValidationError
    with self.assertRaises(pydantic.ValidationError):
      local_connection_config.LocalAgentConfig(
          workspaces=typing.cast(typing.Any, 123)
      )

    with self.assertRaises(pydantic.ValidationError):
      local_connection_config.LocalAgentConfig(
          workspaces=typing.cast(typing.Any, {"invalid": "type"})
      )

  def test_strategy_normalizes_single_workspace_input(self):
    """Verifies that a single string or PathLike passed to strategy is normalized."""
    strategy_str = self._make_strategy(workspaces="ws")
    self.assertEqual(
        strategy_str._workspaces, [str(pathlib.Path("ws").resolve())]
    )

    strategy_path = self._make_strategy(workspaces=pathlib.Path("ws"))
    self.assertEqual(
        strategy_path._workspaces, [str(pathlib.Path("ws").resolve())]
    )

  def test_mcp_servers_propagated(self):
    """Verifies that mcp_servers are correctly serialized to HarnessConfig."""
    mcp_servers = [
        types.McpStreamableHttpServer(
            name="my_http_server",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer token123"},
            timeout_seconds=30,
        ),
        types.McpStdioServer(
            name="my_stdio_server",
            command="node",
            args=["server.js"],
            env={"NODE_ENV": "production"},
            timeout_seconds=10,
        ),
    ]
    strategy = self._make_strategy(mcp_servers=mcp_servers)
    config = strategy._build_harness_config()

    self.assertLen(config.mcp_servers, 2)

    # Assert HTTP server fields
    http_server = config.mcp_servers[0]
    self.assertEqual(http_server.name, "my_http_server")
    self.assertTrue(http_server.HasField("http"))
    self.assertEqual(http_server.http.url, "http://localhost:8080/mcp")
    self.assertEqual(
        http_server.http.headers["Authorization"], "Bearer token123"
    )
    self.assertEqual(http_server.timeout_seconds, 30)

    # Assert Stdio server fields
    stdio_server = config.mcp_servers[1]
    self.assertEqual(stdio_server.name, "my_stdio_server")
    self.assertTrue(stdio_server.HasField("stdio"))
    self.assertEqual(stdio_server.stdio.command, "node")
    self.assertEqual(stdio_server.stdio.args, ["server.js"])
    self.assertEqual(stdio_server.stdio.env["NODE_ENV"], "production")
    self.assertEqual(stdio_server.timeout_seconds, 10)

  def test_retry_config_none_or_default(self):
    """Verifies that unset or default retry_config leaves proto field unset."""
    strategy_none = self._make_strategy(retry_config=None)
    config_none = strategy_none._build_harness_config()
    self.assertFalse(config_none.HasField("retry_config"))

    strategy_empty = self._make_strategy(retry_config=types.RetryConfig())
    config_empty = strategy_empty._build_harness_config()
    self.assertFalse(config_empty.HasField("retry_config"))

  def test_budget_config_proto(self):
    """Verifies that budget_config translates to proto correctly."""
    strategy_none = self._make_strategy(budget_config=None)
    config_none = strategy_none._build_harness_config()
    self.assertFalse(config_none.HasField("budget_config"))

    strategy_empty = self._make_strategy(budget_config=types.BudgetConfig())
    config_empty = strategy_empty._build_harness_config()
    self.assertFalse(config_empty.HasField("budget_config"))

    budget_cfg = types.BudgetConfig(
        max_model_calls=5,
        max_tool_calls=10,
        max_input_tokens=500,
        max_output_tokens=200,
        max_total_tokens=1000,
    )
    strategy = self._make_strategy(budget_config=budget_cfg)
    config = strategy._build_harness_config()
    self.assertTrue(config.HasField("budget_config"))
    self.assertEqual(config.budget_config.max_model_calls, 5)
    self.assertEqual(config.budget_config.max_tool_calls, 10)
    self.assertEqual(config.budget_config.max_input_tokens, 500)
    self.assertEqual(config.budget_config.max_output_tokens, 200)
    self.assertEqual(config.budget_config.max_total_tokens, 1000)
    self.assertEqual(
        config.budget_config.scope,
        localharness_pb2.BudgetConfig.BUDGET_SCOPE_LIFETIME,
    )

    budget_cfg_lifetime = types.BudgetConfig(
        max_total_tokens=1000,
        scope=types.BudgetScope.LIFETIME,
    )
    strategy_lifetime = self._make_strategy(budget_config=budget_cfg_lifetime)
    config_lifetime = strategy_lifetime._build_harness_config()
    self.assertTrue(config_lifetime.HasField("budget_config"))
    self.assertEqual(
        config_lifetime.budget_config.scope,
        localharness_pb2.BudgetConfig.BUDGET_SCOPE_LIFETIME,
    )

    budget_cfg_forward = types.BudgetConfig(
        max_total_tokens=2000,
        scope=types.BudgetScope.FORWARD_LOOKING,
    )
    strategy_forward = self._make_strategy(budget_config=budget_cfg_forward)
    config_forward = strategy_forward._build_harness_config()
    self.assertTrue(config_forward.HasField("budget_config"))
    self.assertEqual(config_forward.budget_config.max_total_tokens, 2000)
    self.assertEqual(
        config_forward.budget_config.scope,
        localharness_pb2.BudgetConfig.BUDGET_SCOPE_FORWARD_LOOKING,
    )

    # Test direct helper with normalized inputs
    self.assertIsNone(
        local_connection.build_budget_config_proto(None)
    )
    self.assertIsNone(
        local_connection.build_budget_config_proto(types.BudgetConfig())
    )

  def test_retry_config_api_retry_only(self):
    """Verifies translation when only api_retry is configured."""
    retry_cfg = types.RetryConfig(
        api_retry=types.ModelAPIRetryConfig(
            max_retries=5,
            initial_sleep_duration_ms=500,
            exponential_multiplier=1.5,
            jitter_range=0.1,
        )
    )
    strategy = self._make_strategy(retry_config=retry_cfg)
    config = strategy._build_harness_config()
    self.assertTrue(config.HasField("retry_config"))
    self.assertTrue(config.retry_config.HasField("api_retry"))
    self.assertFalse(config.retry_config.HasField("model_output_retry"))
    self.assertEqual(config.retry_config.api_retry.max_retries, 5)
    self.assertEqual(
        config.retry_config.api_retry.initial_sleep_duration_ms, 500
    )
    self.assertAlmostEqual(
        config.retry_config.api_retry.exponential_multiplier, 1.5
    )
    self.assertAlmostEqual(config.retry_config.api_retry.jitter_range, 0.1)

  def test_retry_config_model_output_retry_only(self):
    """Verifies translation when only model_output_retry is configured."""
    retry_cfg = types.RetryConfig(
        model_output_retry=types.ModelOutputRetryConfig(max_retries=3)
    )
    strategy = self._make_strategy(retry_config=retry_cfg)
    config = strategy._build_harness_config()
    self.assertTrue(config.HasField("retry_config"))
    self.assertFalse(config.retry_config.HasField("api_retry"))
    self.assertTrue(config.retry_config.HasField("model_output_retry"))
    self.assertEqual(config.retry_config.model_output_retry.max_retries, 3)

  def test_retry_config_all_fields_set(self):
    """Verifies translation when both API retry and output retry are set."""
    retry_cfg = types.RetryConfig(
        api_retry=types.ModelAPIRetryConfig(
            max_retries=10,
            initial_sleep_duration_ms=1000,
            exponential_multiplier=2.0,
            jitter_range=0.2,
        ),
        model_output_retry=types.ModelOutputRetryConfig(max_retries=5),
    )
    strategy = self._make_strategy(retry_config=retry_cfg)
    config = strategy._build_harness_config()
    self.assertTrue(config.retry_config.HasField("api_retry"))
    self.assertTrue(config.retry_config.HasField("model_output_retry"))
    self.assertEqual(config.retry_config.api_retry.max_retries, 10)
    self.assertEqual(
        config.retry_config.api_retry.initial_sleep_duration_ms, 1000
    )
    self.assertAlmostEqual(
        config.retry_config.api_retry.exponential_multiplier, 2.0
    )
    self.assertAlmostEqual(config.retry_config.api_retry.jitter_range, 0.2)
    self.assertEqual(config.retry_config.model_output_retry.max_retries, 5)

  def test_retry_config_boundary_and_edge_values(self):
    """Verifies translation of boundary values like 0, max uint32, and float multipliers."""
    max_uint32 = 4294967295
    retry_cfg = types.RetryConfig(
        api_retry=types.ModelAPIRetryConfig(
            max_retries=0,
            initial_sleep_duration_ms=max_uint32,
            exponential_multiplier=0.0001,
            jitter_range=1.0,
        ),
        model_output_retry=types.ModelOutputRetryConfig(max_retries=max_uint32),
    )
    strategy = self._make_strategy(retry_config=retry_cfg)
    config = strategy._build_harness_config()
    self.assertEqual(config.retry_config.api_retry.max_retries, 0)
    self.assertEqual(
        config.retry_config.api_retry.initial_sleep_duration_ms, max_uint32
    )
    self.assertAlmostEqual(
        config.retry_config.api_retry.exponential_multiplier, 0.0001
    )
    self.assertAlmostEqual(config.retry_config.api_retry.jitter_range, 1.0)
    self.assertEqual(
        config.retry_config.model_output_retry.max_retries, max_uint32
    )

  def test_retry_config_from_local_agent_config(self):
    """Verifies end-to-end propagation from LocalAgentConfig through create_strategy."""
    cfg = local_connection_config.LocalAgentConfig(
        retry_config=types.RetryConfig(
            api_retry=types.ModelAPIRetryConfig(
                max_retries=7, initial_sleep_duration_ms=250
            )
        )
    )
    strategy = cfg.create_strategy(tool_runner=None, hook_runner=None)
    config = strategy._build_harness_config()
    self.assertTrue(config.HasField("retry_config"))
    self.assertEqual(config.retry_config.api_retry.max_retries, 7)
    self.assertEqual(
        config.retry_config.api_retry.initial_sleep_duration_ms, 250
    )


class LocalConnectionStrategyApiKeyTest(unittest.IsolatedAsyncioTestCase):
  """Tests for API key validation in LocalConnectionStrategy."""

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)

  def _make_strategy(self, **kwargs):
    """Creates a LocalConnectionStrategy with the given kwargs."""
    return local_connection.LocalConnectionStrategy(**kwargs)

  @mock.patch.dict("os.environ", {}, clear=True)
  async def test_raises_without_api_key(self):
    """Verifies entry raises when no API key is available.

    Why: The Go localharness binary silently returns empty responses when no
    API key is provided. An explicit error at startup is much more actionable.
    How: Create a strategy with a model target and assert
    AntigravityValidationError.
    """
    models = [
        types.ModelTarget(
            name="gemini-3.8-flash",
            types=[types.ModelType.TEXT],
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(types.AntigravityValidationError) as ctx:
      async with strategy:
        pass
    self.assertIn("must have an endpoint configured", str(ctx.exception))

  @mock.patch.dict("os.environ", {}, clear=True)
  async def test_raises_with_empty_endpoint_api_key(self):
    """Verifies entry raises when GeminiAPIEndpoint has no api_key and env is unset.

    Why: GeminiAPIEndpoint(api_key=None) must not be fooled by empty values.
    """
    models = [
        types.ModelTarget(
            name="gemini-3.8-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(api_key=None),
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(types.AntigravityValidationError) as ctx:
      async with strategy:
        pass
    self.assertIn("A Gemini API key is required", str(ctx.exception))

  @mock.patch.dict("os.environ", {}, clear=True)
  async def test_raises_without_auth_in_vertex_mode(self):
    """Verifies strategy raises validation error when Vertex is set but no project/location provided."""
    models = [
        types.ModelTarget(
            name="gemini-3.8-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.VertexEndpoint(project=None, location=None),
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(types.AntigravityValidationError) as ctx:
      async with strategy:
        pass
    self.assertIn(
        "either (project and location) or api_key must be set",
        str(ctx.exception),
    )

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("subprocess.Popen")
  async def test_accepts_vertex_config_with_project_location(self, mock_popen):
    """Verifies entry does not raise when Vertex is enabled and project/location are provided."""
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc

    models = [
        types.ModelTarget(
            name="gemini-3.8-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.VertexEndpoint(
                project="my-project",
                location="us-central1",
            ),
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(RuntimeError):
      async with strategy:
        pass

  @mock.patch.dict(
      "os.environ",
      {
          "GOOGLE_GENAI_USE_VERTEXAI": "True",
          "GOOGLE_CLOUD_PROJECT": "env-project",
          "GOOGLE_CLOUD_LOCATION": "env-location",
      },
      clear=True,
  )
  @mock.patch("subprocess.Popen")
  async def test_bare_config_routes_to_vertex_via_env(self, mock_popen):
    """Bare LocalAgentConfig + Vertex env vars routes to Vertex and validates."""
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc

    cfg = local_connection_config.LocalAgentConfig(model="gemini-3.8-flash")
    self.assertIsInstance(cfg.models[0].endpoint, types.VertexEndpoint)
    self.assertEqual(cfg.models[0].endpoint.project, "env-project")
    self.assertEqual(cfg.models[0].endpoint.location, "env-location")
    strategy = self._make_strategy(models=cfg.models)
    with self.assertRaises(RuntimeError):
      async with strategy:
        pass

  @mock.patch.dict(
      "os.environ", {"GOOGLE_GENAI_USE_ENTERPRISE": "True"}, clear=True
  )
  def test_bare_config_routes_to_vertex_via_use_enterprise_env(self):
    """USE_ENTERPRISE alone also triggers Vertex routing (GEAP recipe)."""
    cfg = local_connection_config.LocalAgentConfig(model="gemini-3.8-flash")
    self.assertIsInstance(cfg.models[0].endpoint, types.VertexEndpoint)

  @mock.patch.dict(
      "os.environ",
      {
          "GOOGLE_CLOUD_PROJECT": "env-project",
          "GOOGLE_CLOUD_LOCATION": "env-location",
      },
      clear=True,
  )
  def test_vertex_endpoint_direct_construction_hydrates_from_env(self):
    """VertexEndpoint() constructed directly also hydrates from env."""
    ep = types.VertexEndpoint()
    self.assertEqual(ep.project, "env-project")
    self.assertEqual(ep.location, "env-location")

  @mock.patch.dict(
      "os.environ",
      {
          "GOOGLE_CLOUD_PROJECT": "env-project",
          "GOOGLE_CLOUD_LOCATION": "env-location",
      },
      clear=True,
  )
  def test_vertex_endpoint_api_key_skips_env_hydration(self):
    """VertexEndpoint(api_key=...) skips hydrating project and location from env."""
    ep = types.VertexEndpoint(api_key="express-key")
    self.assertEqual(ep.api_key, "express-key")
    self.assertIsNone(ep.project)
    self.assertIsNone(ep.location)
    ep.validate_endpoint()

  @mock.patch.dict(
      "os.environ",
      {
          "GOOGLE_CLOUD_PROJECT": "env-project",
          "GOOGLE_CLOUD_LOCATION": "env-location",
      },
      clear=True,
  )
  def test_vertex_endpoint_base_url_skips_env_hydration(self):
    """VertexEndpoint(base_url=...) skips hydrating project and location from env."""
    ep = types.VertexEndpoint(base_url="http://localhost:8080")
    self.assertEqual(ep.base_url, "http://localhost:8080")
    self.assertIsNone(ep.project)
    self.assertIsNone(ep.location)
    self.assertIsNone(ep.api_key)
    ep.validate_endpoint()

  def test_vertex_endpoint_base_url_allows_custom_auth_and_routing(self):
    """VertexEndpoint allows custom project/location and api_key when base_url is set."""
    ep = types.VertexEndpoint(
        base_url="https://gateway.example.com/vertex",
        project="my-proj",
        location="us-central1",
        api_key="gateway-key",
        http_headers={"Authorization": "Bearer token"},
    )
    self.assertEqual(ep.base_url, "https://gateway.example.com/vertex")
    self.assertEqual(ep.project, "my-proj")
    self.assertEqual(ep.location, "us-central1")
    self.assertEqual(ep.api_key, "gateway-key")
    ep.validate_endpoint()

  def test_vertex_endpoint_mutual_exclusivity(self):
    """VertexEndpoint raises ValueError when both api_key and project/location are set without base_url."""
    ep_both = types.VertexEndpoint(
        project="my-proj", location="us-central1", api_key="express-key"
    )
    with self.assertRaisesRegex(ValueError, "Cannot specify both api_key"):
      ep_both.validate_endpoint()

    ep_proj = types.VertexEndpoint(project="my-proj", api_key="express-key")
    with self.assertRaisesRegex(ValueError, "Cannot specify both api_key"):
      ep_proj.validate_endpoint()

    ep_loc = types.VertexEndpoint(location="us-central1", api_key="express-key")
    with self.assertRaisesRegex(ValueError, "Cannot specify both api_key"):
      ep_loc.validate_endpoint()

  @mock.patch.dict("os.environ", {}, clear=True)
  def test_vertex_shorthand_forwards_api_key(self):
    """LocalAgentConfig(vertex=True, api_key=...) constructs Express mode VertexEndpoint."""
    cfg = local_connection_config.LocalAgentConfig(
        vertex=True, api_key="express-key"
    )
    self.assertIsInstance(cfg.models[0].endpoint, types.VertexEndpoint)
    self.assertEqual(cfg.models[0].endpoint.api_key, "express-key")
    self.assertIsNone(cfg.models[0].endpoint.project)
    self.assertIsNone(cfg.models[0].endpoint.location)
    cfg.models[0].endpoint.validate_endpoint()

  @mock.patch.dict("os.environ", {}, clear=True)
  def test_vertex_express_config_propagates_to_harness_proto(self):
    """Verifies that api_key on VertexEndpoint propagates to localharness proto."""
    models = [
        types.ModelTarget(
            name="gemini-3.8-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.VertexEndpoint(api_key="express-key"),
        )
    ]
    strategy = self._make_strategy(models=models)
    config_proto = strategy._build_harness_config()
    self.assertEqual(
        config_proto.models[0].vertex_endpoint.api_key, "express-key"
    )

  @mock.patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}, clear=True)
  @mock.patch("subprocess.Popen")
  async def test_accepts_env_var_api_key(self, mock_popen):
    """Verifies entry does not raise when GEMINI_API_KEY env var is set.

    Why: The env var fallback is the most common path for 3P developers.
    How: Set GEMINI_API_KEY, enter the context manager, and verify it proceeds
    past the validation check (it will fail later at subprocess I/O, which is
    expected).

    Args:
      mock_popen: Mocked subprocess.Popen to prevent actual process launch.
    """
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc
    strategy = self._make_strategy()
    # Should not raise AntigravityValidationError; it will raise RuntimeError
    # from the subprocess read failure, which proves we passed the check.
    with self.assertRaises(RuntimeError):
      async with strategy:
        pass

  @mock.patch.dict(
      "os.environ",
      {"GEMINI_API_KEY": "env-key", "SYS_VAR": "sys_val"},
      clear=True,
  )
  @mock.patch("subprocess.Popen")
  async def test_passes_custom_env_to_popen_and_input_config(self, mock_popen):
    """Verifies custom env dict is merged into Popen env and InputConfig."""
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc

    custom_env = {"MY_CUSTOM_VAR": "hello_env"}
    strategy = self._make_strategy(env=custom_env)

    with self.assertRaises(RuntimeError):
      async with strategy:
        pass

    mock_popen.assert_called_once()
    _, kwargs = mock_popen.call_args
    expected_env = {
        "GEMINI_API_KEY": "env-key",
        "SYS_VAR": "sys_val",
        "MY_CUSTOM_VAR": "hello_env",
    }
    self.assertEqual(kwargs.get("env"), expected_env)

    mock_proc.stdin.write.assert_called_once()
    written_bytes = mock_proc.stdin.write.call_args[0][0]
    parsed_config = localharness_pb2.InputConfig()
    parsed_config.ParseFromString(written_bytes[4:])
    self.assertEqual(dict(parsed_config.env), custom_env)

  @mock.patch.dict(
      "os.environ",
      {"GEMINI_API_KEY": "env-key"},
      clear=True,
  )
  @mock.patch("subprocess.Popen")
  async def test_passes_non_string_env_coerced_to_strings(self, mock_popen):
    """Verifies non-string env keys/values are coerced to strings for Popen and InputConfig."""
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc

    custom_env = {"INT_KEY": 123, 1: "val", "BOOL_KEY": True}
    strategy = self._make_strategy(env=custom_env)

    with self.assertRaises(RuntimeError):
      async with strategy:
        pass

    mock_popen.assert_called_once()
    _, kwargs = mock_popen.call_args
    expected_env = {
        "GEMINI_API_KEY": "env-key",
        "INT_KEY": "123",
        "1": "val",
        "BOOL_KEY": "True",
    }
    self.assertEqual(kwargs.get("env"), expected_env)

    mock_proc.stdin.write.assert_called_once()
    written_bytes = mock_proc.stdin.write.call_args[0][0]
    parsed_config = localharness_pb2.InputConfig()
    parsed_config.ParseFromString(written_bytes[4:])
    self.assertEqual(
        dict(parsed_config.env),
        {"INT_KEY": "123", "1": "val", "BOOL_KEY": "True"},
    )

  def test_config_env_defaults_to_none(self):
    """Verifies that LocalAgentConfig.env is None by default."""
    config = local_connection_config.LocalAgentConfig()
    self.assertIsNone(config.env)

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("subprocess.Popen")
  async def test_accepts_models_api_key(self, mock_popen):
    """Verifies entry does not raise when model endpoint api_key is set.

    Why: Explicit API key in config is the recommended path.
    How: Set api_key in GeminiAPIEndpoint, enter context manager.

    Args:
      mock_popen: Mocked subprocess.Popen to prevent actual process launch.
    """
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc
    models = [
        types.ModelTarget(
            name="gemini-3.8-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(api_key="explicit-key"),
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(RuntimeError):
      async with strategy:
        pass


class LocalConnectionStrategyConnectTest(unittest.IsolatedAsyncioTestCase):
  """Tests for WebSocket connection fallback in LocalConnectionStrategy."""

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)

  def _make_strategy(self, **kwargs):
    return local_connection.LocalConnectionStrategy(**kwargs)

  @mock.patch("websockets.connect", new_callable=mock.AsyncMock)
  @mock.patch("subprocess.Popen")
  async def test_connect_falls_back_to_127_0_0_1_on_localhost_failure(
      self, mock_popen, mock_connect
  ):
    """Verifies that if localhost fails, strategy attempts connecting via 127.0.0.1."""
    output_config = localharness_pb2.OutputConfig(port=8080, api_key="fake-key")
    serialized = output_config.SerializeToString()
    length_bytes = struct.pack("<I", len(serialized))

    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = io.BytesIO(b"")
    mock_proc.stdout.read.side_effect = [length_bytes, serialized]
    mock_popen.return_value = mock_proc

    mock_ws = mock.MagicMock()
    mock_ws.send = mock.AsyncMock()
    mock_ws.recv = mock.AsyncMock(return_value="{}")
    mock_ws.close = mock.AsyncMock()
    mock_ws.__aiter__.return_value = []

    mock_connect.side_effect = [
        OSError("localhost resolution failed"),
        mock_ws,
    ]

    models = [
        types.ModelTarget(
            name="gemini-3.5-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(api_key="explicit-key"),
        )
    ]
    strategy = self._make_strategy(models=models)

    async with strategy:
      pass

    self.assertEqual(mock_connect.call_count, 2)
    mock_connect.assert_has_calls([
        mock.call(
            "ws://localhost:8080/",
            additional_headers={"x-goog-api-key": "fake-key"},
            max_size=None,
        ),
        mock.call(
            "ws://127.0.0.1:8080/",
            additional_headers={"x-goog-api-key": "fake-key"},
            max_size=None,
        ),
    ])


_get_default_binary_path = local_connection._get_default_binary_path


class GetDefaultBinaryPathTest(unittest.TestCase):

  @mock.patch.dict(
      "os.environ",
      {"ANTIGRAVITY_HARNESS_PATH": "/I/should/not/be/used"},
  )
  def test_returns_env_path_via_passed_env(self):
    path = _get_default_binary_path(
        env={"ANTIGRAVITY_HARNESS_PATH": "/I/should/be/used"}
    )
    self.assertEqual(path, "/I/should/be/used")

  @mock.patch.dict(
      "os.environ",
      {"ANTIGRAVITY_HARNESS_PATH": "/env/path"}
  )
  def test_returns_env_path_via_global_env(self):
    path = _get_default_binary_path(env={"not_harness_path": "different_value"})
    self.assertEqual(path, "/env/path")

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("importlib.metadata.distribution")
  @mock.patch("os.path.exists")
  def test_returns_metadata_distribution_path(self, mock_exists, mock_dist):
    mock_file = mock.MagicMock()
    mock_file.__str__.return_value = "google/antigravity/bin/localharness"
    mock_file.locate.return_value = (
        "/site-packages/google/antigravity/bin/localharness"
    )

    mock_distribution = mock.MagicMock()
    mock_distribution.files = [mock_file]
    mock_dist.return_value = mock_distribution
    mock_exists.return_value = True

    path = _get_default_binary_path(env=None)
    self.assertEqual(path, "/site-packages/google/antigravity/bin/localharness")
    mock_dist.assert_called_once_with("google-antigravity")
    mock_file.locate.assert_called_once()

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("importlib.metadata.distribution")
  @mock.patch("importlib.resources.files")
  @mock.patch("os.path.exists")
  def test_returns_external_wheel_path(
      self, mock_exists, mock_files, mock_dist
  ):
    mock_dist.side_effect = importlib.metadata.PackageNotFoundError
    mock_path = mock.MagicMock()
    mock_path.joinpath.return_value.__str__.return_value = "/wheel/path"
    mock_files.return_value = mock_path
    mock_exists.return_value = True

    path = _get_default_binary_path(env={})
    self.assertEqual(path, "/wheel/path")

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("importlib.metadata.distribution")
  @mock.patch("importlib.resources.files")
  @mock.patch("shutil.which")
  def test_returns_system_path(self, mock_which, mock_files, mock_dist):
    mock_dist.side_effect = importlib.metadata.PackageNotFoundError
    mock_files.side_effect = ImportError
    mock_which.return_value = "/system/path"

    path = _get_default_binary_path(env={})
    self.assertEqual(path, "/system/path")
    mock_which.assert_called_once_with("localharness")

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("importlib.metadata.distribution")
  @mock.patch("importlib.resources.files")
  @mock.patch("shutil.which")
  def test_raises_when_not_found(self, mock_which, mock_files, mock_dist):
    mock_dist.side_effect = importlib.metadata.PackageNotFoundError
    mock_files.side_effect = ImportError
    mock_which.return_value = None

    with self.assertRaises(RuntimeError) as ctx:
      _get_default_binary_path(env={})
    self.assertIn(
        "Could not find default localharness binary", str(ctx.exception)
    )


class LocalConnectionSessionHooksTest(unittest.IsolatedAsyncioTestCase):
  """Tests for session start/end hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.tool_runner = tool_runner.ToolRunner()

  async def test_strategy_dispatches_session_start(self):
    """Verifies the connection correctly delegates OnSessionStart hook execution to the HookRunner whenever requested by the harness engine."""
    called = []
    event = asyncio.Event()

    class SessionStartHook(hooks_base.OnSessionStartHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        called.append("started")
        event.set()

    hr = hook_runner.HookRunner(on_session_start_hooks=[SessionStartHook()])

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    req = localharness_pb2.CallHookRequest(
        request_id="req_1",
        name="OnSessionStart",
        type=localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_START,
    )
    await harness.send_event(
        localharness_pb2.OutputEvent(call_hook_request=req)
    )
    await asyncio.wait_for(event.wait(), timeout=1.0)
    self.assertEqual(called, ["started"])

  async def test_session_end_hook_dispatched_on_disconnect(self):
    """Verifies OnSessionEndHook fires via LSP handshake when disconnect() is called."""
    called = []
    event = asyncio.Event()

    class SessionEndHook(hooks_base.OnSessionEndHook):

      async def run(self, context: hooks_base.HookContext, data: None):  # pylint: disable=unused-argument
        called.append("ended")
        event.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(SessionEndHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    disconnect_task = asyncio.create_task(harness.conn.disconnect())

    # 1. SDK emits session_end_request
    resp = await harness.wait_for_response()
    self.assertTrue(resp.get("sessionEndRequest"))

    # 2. Simulate Go harness sending CallHookRequest(OnSessionEnd)
    req = localharness_pb2.CallHookRequest(
        request_id="req_end",
        name="OnSessionEnd",
        type=localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_END,
    )
    await harness.send_event(
        localharness_pb2.OutputEvent(call_hook_request=req)
    )

    # 3. SDK routes request and replies CallHookResponse
    hook_resp = await harness.wait_for_response()
    self.assertIn("callHookResponse", hook_resp)
    self.assertEqual(hook_resp["callHookResponse"]["requestId"], "req_end")

    # 4. Simulate Go harness sending SessionEndResponse
    await harness.send_event(
        localharness_pb2.OutputEvent(session_end_response=True)
    )

    await asyncio.wait_for(disconnect_task, timeout=1.0)
    self.assertEqual(called, ["ended"])


class LocalConnectionPostTurnHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for post-turn hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.tool_runner = tool_runner.ToolRunner()

  async def test_post_turn_hook_dispatched_on_final_step(self):
    """Verifies PostTurnHook fires when a terminal model step is received."""
    captured = []

    class PostTurnHook(hooks_base.PostTurnHook):

      async def run(self, context: hooks_base.HookContext, data: str):  # pylint: disable=unused-argument
        captured.append(data)

    hr = hook_runner.HookRunner()
    hr.register_hook(PostTurnHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    # Simulate a send to create turn context.
    await harness.conn.send("hello")

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=1,
            text="Final answer",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_USER,
        )
    )

    await harness.send_event(event)

    # The real harness sends STATE_IDLE after the final step. The
    # connection waits for this before returning from receive_steps().
    idle_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="test_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
        )
    )
    await harness.send_event(idle_event)
    await harness.send_event(
        localharness_pb2.OutputEvent(
            call_hook_request=localharness_pb2.CallHookRequest(
                request_id="post_req_1",
                name="PostTurn",
                type=localharness_pb2.LIFECYCLE_HOOK_POST_TURN,
                post_turn_args=localharness_pb2.PostTurnArgs(
                    response_text="Final answer"
                ),
            )
        )
    )
    await asyncio.sleep(0.05)

    # Drain receive_steps to trigger terminal detection + hook dispatch.
    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(captured, ["Final answer"])

  async def test_receive_steps_includes_target_environment(self):
    """Verifies TARGET_ENVIRONMENT steps are yielded by receive_steps()."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

    # Simulate a send to create turn context.
    await harness.conn.send("hello")

    # Step 1: A TARGET_ENVIRONMENT step (tool execution).
    env_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=1,
            text="Requesting permission to make tool call",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_ENVIRONMENT,
        )
    )

    # Step 2: A TARGET_USER step (the final answer).
    user_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=2,
            text="Here is my answer.",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_USER,
        )
    )

    idle_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="test_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
        )
    )

    await harness.send_event(env_event)
    await harness.send_event(user_event)
    await harness.send_event(idle_event)

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    # Both steps must be yielded (the old filter would have dropped step 1).
    self.assertEqual(len(steps), 2)

    # Step 1: environment step — yielded but NOT a final response.
    self.assertEqual(
        steps[0].content, "Requesting permission to make tool call"
    )
    self.assertEqual(steps[0].target, "TARGET_ENVIRONMENT")
    self.assertFalse(steps[0].is_complete_response)

    # Step 2: user step — the real final response.
    self.assertEqual(steps[1].content, "Here is my answer.")
    self.assertEqual(steps[1].target, "TARGET_USER")
    self.assertTrue(steps[1].is_complete_response)

  async def test_post_turn_hook_not_fired_for_environment_step(self):
    """Verifies PostTurnHook does NOT fire for TARGET_ENVIRONMENT steps."""
    captured = []

    class PostTurnHook(hooks_base.PostTurnHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured.append(data)

    hr = hook_runner.HookRunner()
    hr.register_hook(PostTurnHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await harness.conn.send("hello")

    # A terminal environment step that should NOT trigger the hook.
    env_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=1,
            text="Requesting permission to make tool call",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_ENVIRONMENT,
        )
    )

    # The real final response that SHOULD trigger the hook.
    user_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=2,
            text="Final answer",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_USER,
        )
    )

    idle_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="test_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_FULLY_IDLE,
        )
    )

    await harness.send_event(env_event)
    await harness.send_event(user_event)
    await harness.send_event(idle_event)
    await harness.send_event(
        localharness_pb2.OutputEvent(
            call_hook_request=localharness_pb2.CallHookRequest(
                request_id="post_req_2",
                name="PostTurn",
                type=localharness_pb2.LIFECYCLE_HOOK_POST_TURN,
                post_turn_args=localharness_pb2.PostTurnArgs(
                    response_text="Final answer"
                ),
            )
        )
    )
    await asyncio.sleep(0.05)

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    # Both steps yielded.
    self.assertEqual(len(steps), 2)

    # Hook fired exactly once, with the TARGET_USER step's content.
    self.assertEqual(captured, ["Final answer"])


class LocalConnectionCompactionHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for compaction hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()

  async def test_compaction_step_dispatches_hook(self):
    """Verifies OnCompactionHook fires when a compaction step is received."""
    captured = []
    event = asyncio.Event()

    class CompactionHook(hooks_base.OnCompactionHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured.append(data)
        if len(captured) == 1:
          event.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(CompactionHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    req = localharness_pb2.CallHookRequest(
        request_id="req_comp_1",
        name="OnCompaction",
        type=localharness_pb2.LIFECYCLE_HOOK_ON_COMPACTION,
        on_compaction_args=localharness_pb2.OnCompactionArgs(
            trajectory_id="main",
            step_index=1,
            summary="Context compaction",
        ),
    )
    output_event = localharness_pb2.OutputEvent(call_hook_request=req)
    await harness.send_event(output_event)

    await asyncio.wait_for(event.wait(), timeout=1.0)
    self.assertEqual(len(captured), 1)
    self.assertEqual(captured[0].type, types.StepType.COMPACTION)
    self.assertEqual(captured[0].content, "Context compaction")
    self.assertEqual(captured[0].status, types.StepStatus.DONE)
    self.assertEqual(captured[0].source, types.StepSource.SYSTEM)
    self.assertEqual(captured[0].target, types.StepTarget.USER)
    self.assertEqual(captured[0].trajectory_id, "main")
    self.assertEqual(captured[0].step_index, 1)

    resp = await harness.wait_for_response(timeout=1.0)
    self.assertIn("callHookResponse", resp)
    self.assertEqual(resp["callHookResponse"]["requestId"], "req_comp_1")
    self.assertIn("emptyResult", resp["callHookResponse"])

  def test_get_enabled_hooks_includes_on_compaction(self):
    """Verifies _get_enabled_hooks includes LIFECYCLE_HOOK_ON_COMPACTION."""
    hr = hook_runner.HookRunner()

    class CompactionHook(hooks_base.OnCompactionHook):

      async def run(self, context, data):
        pass

    hr.register_hook(CompactionHook())
    strategy = local_connection.LocalConnectionStrategy(hook_runner=hr)
    enabled = strategy._get_enabled_hooks()
    self.assertIn(localharness_pb2.LIFECYCLE_HOOK_ON_COMPACTION, enabled)


class LocalConnectionStopHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for Stop lifecycle hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()

  def test_get_enabled_hooks_includes_stop(self):
    """Verifies _get_enabled_hooks includes LIFECYCLE_HOOK_STOP."""
    hr = hook_runner.HookRunner()

    class StopHook(hooks_base.StopHook):

      async def run(self, context, data):
        return types.StopHookResult()

    hr.register_hook(StopHook())
    strategy = local_connection.LocalConnectionStrategy(hook_runner=hr)
    enabled = strategy._get_enabled_hooks()
    self.assertIn(localharness_pb2.LIFECYCLE_HOOK_STOP, enabled)

  async def test_stop_hook_dispatched_via_local_harness(self):
    """Verifies StopHook fires and returns StopResult over the wire."""
    captured = []
    event = asyncio.Event()

    class CustomStopHook(hooks_base.StopHook):

      async def run(self, context, data):
        captured.append(data)
        event.set()
        return types.StopHookResult(
            decision=types.StopDecision.CONTINUE,
            reason="Continue working",
        )

    hr = hook_runner.HookRunner()
    hr.register_hook(CustomStopHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    req = localharness_pb2.CallHookRequest(
        request_id="req_stop_1",
        name="Stop",
        type=localharness_pb2.LIFECYCLE_HOOK_STOP,
        stop_args=localharness_pb2.StopArgs(
            response_text="Done with task",
            trajectory_id="main_traj",
            continuation_count=0,
            stop_reason=localharness_pb2.TrajectoryStateUpdate.STOP_REASON_UNSPECIFIED,
            error_message="",
        ),
    )
    output_event = localharness_pb2.OutputEvent(call_hook_request=req)
    await harness.send_event(output_event)

    await asyncio.wait_for(event.wait(), timeout=1.0)
    self.assertEqual(len(captured), 1)
    self.assertEqual(captured[0].response_text, "Done with task")
    self.assertEqual(captured[0].trajectory_id, "main_traj")
    self.assertEqual(captured[0].continuation_count, 0)
    self.assertEqual(captured[0].stop_reason, types.StopReason.UNSPECIFIED)

    resp = await harness.wait_for_response(timeout=1.0)
    self.assertIn("callHookResponse", resp)
    self.assertEqual(resp["callHookResponse"]["requestId"], "req_stop_1")
    self.assertIn("stopResult", resp["callHookResponse"])
    self.assertEqual(
        resp["callHookResponse"]["stopResult"]["decision"],
        "CONTINUE",
    )
    self.assertEqual(
        resp["callHookResponse"]["stopResult"]["reason"],
        "Continue working",
    )


class LocalConnectionSubagentHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for subagent hook dispatch via tool hooks.

  Subagent invocations are treated as tool calls with the name
  START_SUBAGENT. Pre- and post-tool-call hooks receive the subagent
  data using standard tool hook dispatch.
  """

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_invoke_subagent_step_classified_as_tool_call(self):
    """Verifies invoke_subagent steps are classified as TOOL_CALL."""
    hr = hook_runner.HookRunner()

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    await harness.conn.send("hello")

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main",
            trajectory_id="main",
            step_index=1,
            text="Invoking subagent",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            invoke_subagent=localharness_pb2.ActionInvokeSubagent(),
        )
    )

    await harness.send_event(event)

    # Drain the queue to inspect the step.
    step = await asyncio.wait_for(harness.conn._step_queue.get(), timeout=2.0)
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

  async def test_ws_reader_parses_usage_update(self):
    """Verifies that _ws_reader_loop parses usage_update events."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    event = localharness_pb2.OutputEvent(
        usage_update=localharness_pb2.UsageUpdate(
            total=localharness_pb2.UsageMetadata(
                prompt_token_count=150,
                cached_content_token_count=50,
                candidates_token_count=75,
                thoughts_token_count=25,
                total_token_count=250,
            ),
        ),
    )
    await harness.send_event(event)

    # Send a step event so we can synchronize on the queue
    step_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main",
            trajectory_id="main",
            step_index=1,
            text="response after usage",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        ),
    )
    await harness.send_event(step_event)
    await asyncio.wait_for(harness.conn._step_queue.get(), timeout=1.0)

    self.assertEqual(
        harness.conn.cumulative_usage,
        types.UsageMetadata(
            prompt_token_count=150,
            cached_content_token_count=50,
            candidates_token_count=75,
            thoughts_token_count=25,
            total_token_count=250,
        ),
    )


class LocalConnectionToolCallHooksTest(unittest.IsolatedAsyncioTestCase):
  """Tests for post-tool-call and on-tool-error hooks."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_on_tool_error_sends_error_to_harness(self):
    """Verifies that tool errors are sent back to the harness without recovery.

    OnToolError dispatch and recovery have been migrated to the Go harness
    (dispatched via HookRouter). The Python SDK sends the error JSON as-is.
    """
    tr = tool_runner.ToolRunner()

    async def failing_handler(**_):
      raise RuntimeError("Intentional failure")

    tr.register(failing_handler, "failing_tool")

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=tr,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_fail",
            name="failing_tool",
            arguments_json="{}",
        )
    )

    await harness.send_event(event)

    # The error should be sent back as-is (no Python-side recovery).
    sent_data = await harness.wait_for_response()
    self.assertIn("toolResponse", sent_data)
    self.assertIn(
        "Intentional failure", sent_data["toolResponse"]["errorMessage"]
    )


class LocalConnectionBuiltinDecideHookTest(unittest.IsolatedAsyncioTestCase):
  """Verifies Decide hooks run for built-in tool confirmations."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_tool_confirmation_always_accepts(self):
    """After migration, _handle_tool_confirmation_request always auto-accepts.

    Pre-tool hooks are now dispatched via Go's FirePreToolHook ->
    CallHookRequest -> HookRouter._handle_pre_tool. The legacy
    ToolConfirmation path only fires when no hooks are registered.
    """

    class DenyAll(hooks_base.PreToolCallDecideHook):

      async def run(self, context, data):
        return hooks_base.HookResult(allow=False, message="Denied")

    hr = hook_runner.HookRunner(pre_tool_call_decide_hooks=[DenyAll()])
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="traj",
            trajectory_id="traj",
            step_index=0,
            text='Requesting permission to call tool "run_command"',
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_ENVIRONMENT,
            tool_confirmation_request=(
                localharness_pb2.ToolConfirmationRequest()
            ),
            run_command=localharness_pb2.ActionRunCommand(
                command_line="rm -rf /",
            ),
        )
    )
    await harness.send_event(event)

    sent = await harness.wait_for_response()
    # Now auto-accepts — hooks dispatch happens via HookRouter, not here.
    self.assertTrue(sent["toolConfirmation"]["accepted"])


class LocalConnectionHookAcceptanceTest(unittest.IsolatedAsyncioTestCase):
  """Verifies that previously-unsupported hooks are now accepted."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_subagent_tool_hooks_accepted(self):
    """Subagent lifecycle is handled by tool hooks; no special subagent lists."""

    class DummyHook(hooks_base.PreToolCallDecideHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        return hooks_base.HookResult(allow=True)

    hr = hook_runner.HookRunner()
    hr.register_hook(DummyHook())

    # Should NOT raise.
    test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

  async def test_compaction_hooks_no_longer_raise(self):
    """Compaction hooks should be accepted now."""

    class DummyHook(hooks_base.OnCompactionHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        pass

    hr = hook_runner.HookRunner()
    hr.register_hook(DummyHook())

    # Should NOT raise.
    test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )


class LocalConnectionStderrReaderTest(unittest.IsolatedAsyncioTestCase):
  """Tests for the background stderr reader thread."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_start_stderr_reader_drains_lines(self):
    """Verifies that _start_stderr_reader captures stderr lines.

    Why: The Go harness writes diagnostic messages to stderr.  If the
    pipe buffer fills, the harness blocks and cannot save trajectory state
    at shutdown.  The reader thread prevents this by draining continuously.
    How: Write lines to a pipe, start the reader, and assert the deque
    contains all written lines.
    """

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    stream = io.BytesIO(b"line1\nline2\nline3\n")
    harness.conn._start_stderr_reader(stream)
    harness.conn._stderr_thread.join(timeout=2)

    self.assertEqual(
        list(harness.conn._stderr_lines), ["line1", "line2", "line3"]
    )

  async def test_stderr_reader_respects_maxlen(self):
    """Verifies the deque drops old lines when it exceeds maxlen.

    Why: Unbounded buffering could consume excessive memory during
    long-running sessions.  The deque is bounded at 100 lines.
    How: Write 105 lines and confirm only the last 100 remain.
    """

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    lines = "".join(f"line{i}\n" for i in range(105))
    stream = io.BytesIO(lines.encode())
    harness.conn._start_stderr_reader(stream)
    harness.conn._stderr_thread.join(timeout=2)

    self.assertEqual(len(harness.conn._stderr_lines), 100)
    self.assertEqual(harness.conn._stderr_lines[0], "line5")
    self.assertEqual(harness.conn._stderr_lines[-1], "line104")

  async def test_stderr_reader_handles_closed_stream(self):
    """Verifies the reader thread exits cleanly when the stream closes.

    Why: On process exit the stderr pipe closes.  The thread must not
    crash or log errors; it should simply stop.
    How: Pass an already-closed stream and verify the thread exits without
    raising.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    stream = io.BytesIO(b"")
    harness.conn._start_stderr_reader(stream)
    harness.conn._stderr_thread.join(timeout=2)
    self.assertFalse(harness.conn._stderr_thread.is_alive())

  async def test_stderr_reader_thread_is_daemon(self):
    """Verifies the stderr reader thread is a daemon thread.

    Why: The stderr reader must not prevent process exit.  If it were a
    non-daemon thread, a hung harness could keep the Python process alive
    indefinitely.
    How: Start the reader and check the thread's daemon attribute.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    stream = io.BytesIO(b"line1\n")
    harness.conn._start_stderr_reader(stream)
    self.assertTrue(harness.conn._stderr_thread.daemon)
    harness.conn._stderr_thread.join(timeout=2)


class LocalConnectionDisconnectTest(unittest.IsolatedAsyncioTestCase):
  """Tests for the disconnect shutdown sequence."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_process.stdin = mock.MagicMock()
    self.mock_process.wait.return_value = 0
    self.mock_ws = test_utils.TestWebSocket()

  async def test_disconnect_sets_disconnecting_flag(self):
    """Verifies _disconnecting is set before any cleanup runs.

    Why: The reader loop uses this flag to distinguish expected closures
    from harness crashes.  It must be set early in disconnect().
    How: Call disconnect and check the flag is True.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.assertTrue(harness.conn._disconnecting)

  async def test_disconnect_closes_stdin(self):
    """Verifies stdin is closed during disconnect to trigger harness save.

    Why: The Go harness monitors stdin for EOF.  On EOF it runs
    cleanupAllAgents which persists trajectory state to disk.  Without
    closing stdin, the trajectory is never saved.
    How: Call disconnect and verify stdin.close() was called.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.mock_process.stdin.close.assert_called_once()

  @mock.patch.object(local_connection, "_PROCESS_WAIT_TIMEOUT_SECONDS", 5)
  async def test_disconnect_waits_for_process(self):
    """Verifies disconnect waits for the harness process to exit.

    Why: The harness needs time to flush trajectory state after stdin
    closes.  Killing it immediately would lose the trajectory.
    How: Call disconnect and verify process.wait(timeout=5) was called.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.mock_process.wait.assert_called_with(timeout=5)

  @mock.patch.object(local_connection, "_PROCESS_WAIT_TIMEOUT_SECONDS", 5)
  async def test_disconnect_terminates_on_timeout(self):
    """Verifies SIGTERM is sent when the process doesn't exit in time.

    Why: If the harness hangs during cleanup, the SDK must not block
    indefinitely.  SIGTERM is the first escalation.
    How: Make wait() raise TimeoutExpired on the first call, then verify
    terminate() is called.
    """
    self.mock_process.wait.side_effect = [
        subprocess.TimeoutExpired("cmd", 5),  # First wait times out.
        0,  # After terminate, process exits.
    ]
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.mock_process.terminate.assert_called_once()

  @mock.patch.object(local_connection, "_PROCESS_WAIT_TIMEOUT_SECONDS", 5)
  async def test_disconnect_kills_on_double_timeout(self):
    """Verifies SIGKILL is sent when SIGTERM also fails.

    Why: If the process ignores SIGTERM, SIGKILL is the last resort.
    How: Make wait() raise TimeoutExpired twice, then verify kill() is called.
    """
    self.mock_process.wait.side_effect = [
        subprocess.TimeoutExpired("cmd", 5),  # First wait.
        subprocess.TimeoutExpired("cmd", 15),  # After terminate.
        0,  # After kill.
    ]
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.mock_process.terminate.assert_called_once()
    self.mock_process.kill.assert_called_once()

  async def test_disconnect_closes_ws_before_stdin(self):
    """Verifies the WebSocket is closed before stdin.

    Why: The Go HTTP handler's defer saves the trajectory when the handler
    returns.  agent.Close() blocks on <-runChan, which requires the Run
    goroutine to exit.  Run exits when the WS input loop breaks.  So the
    WS must close first to unblock agent.Close().  Stdin close triggers
    os.Exit(0), so it must come after the defer has had time to save.
    How: Record the call order of ws.close and stdin.close.
    """
    call_order = []

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    original_close = harness.ws.close

    async def track_ws_close():
      call_order.append("ws_close")
      await original_close()

    harness.ws.close = track_ws_close
    self.mock_process.stdin.close.side_effect = lambda: call_order.append(
        "stdin_close"
    )

    await harness.disconnect_sdk()
    self.assertEqual(call_order, ["ws_close", "stdin_close"])


class LocalConnectionUnexpectedCloseTest(unittest.IsolatedAsyncioTestCase):
  """Tests for error surfacing when the harness crashes mid-session."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()

  async def test_unexpected_ws_close_surfaces_stderr(self):
    """Verifies harness stderr is surfaced when the WS closes unexpectedly.

    Why: When the harness crashes (e.g., model error, OOM), the WebSocket
    closes with code 1006.  The user needs the harness stderr to diagnose
    the failure.  Previously, this was silently logged and swallowed.
    How: Simulate a ConnectionClosed exception in the reader loop and
    verify an AntigravityConnectionError with stderr content is queued.
    """

    # Create a FakeWebSocket that raises ConnectionClosed immediately.
    class CrashingWebSocket:

      def __init__(self):
        self.sent_messages = []

      async def send(self, message):
        self.sent_messages.append(message)

      def __aiter__(self):
        async def _gen():
          raise websockets.ConnectionClosed(rcvd=None, sent=None)
          yield  # Make it a generator.  pylint: disable=unreachable

        return _gen()

      async def close(self):
        pass

    ws = CrashingWebSocket()
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=ws,
    )
    # Seed some stderr context.
    conn._stderr_lines.append("Failed to call model: quota exceeded")

    # The step queue should contain the error, then the sentinel None.
    item = await asyncio.wait_for(conn._step_queue.get(), timeout=2)
    self.assertIsInstance(item, types.AntigravityConnectionError)
    self.assertIn("quota exceeded", str(item))
    self.assertIn("WS close code", str(item))

  async def test_expected_ws_close_does_not_surface_error(self):
    """Verifies no error is queued when disconnect() initiated the close.

    Why: When the user calls disconnect(), the WebSocket close is expected
    and should not be reported as an error.
    How: Set _disconnecting=True, trigger a ConnectionClosed, and verify
    only the sentinel (None) is in the queue.
    """

    class DisconnectingWebSocket:

      def __init__(self):
        self.sent_messages = []

      async def send(self, message):
        self.sent_messages.append(message)

      def __aiter__(self):
        async def _gen():
          raise websockets.ConnectionClosed(rcvd=None, sent=None)
          yield  # pylint: disable=unreachable

        return _gen()

      async def close(self):
        pass

    ws = DisconnectingWebSocket()
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=ws,
    )
    conn._disconnecting = True

    # Should only see the sentinel, not an error.
    item = await asyncio.wait_for(conn._step_queue.get(), timeout=2)
    self.assertIsNone(item)

  def test_get_ws_close_code_with_rcvd_code(self):
    """Verifies close code extraction when rcvd has a code."""
    rcvd_frame = mock.MagicMock(code=1000)
    exc = websockets.ConnectionClosed(rcvd=rcvd_frame, sent=None)
    self.assertEqual(local_connection._get_ws_close_code(exc), 1000)

  def test_get_ws_close_code_with_sent_code(self):
    """Verifies close code extraction when sent has a code."""
    sent_frame = mock.MagicMock(code=1001)
    exc = websockets.ConnectionClosed(rcvd=None, sent=sent_frame)
    self.assertEqual(local_connection._get_ws_close_code(exc), 1001)

  def test_get_ws_close_code_with_both_none(self):
    """Verifies close code defaults to 1006 when both rcvd and sent are None."""
    exc = websockets.ConnectionClosed(rcvd=None, sent=None)
    self.assertEqual(local_connection._get_ws_close_code(exc), 1006)

  def test_get_ws_close_code_fallback_to_code_attr(self):
    """Verifies fallback to code attribute if rcvd/sent are not None but have no code attribute."""
    exc = mock.MagicMock(spec=websockets.ConnectionClosed)
    exc.rcvd = "invalid_rcvd_no_code"
    exc.sent = "invalid_sent_no_code"
    exc.code = 1008
    self.assertEqual(local_connection._get_ws_close_code(exc), 1008)


class LocalConnectionSendTest(unittest.IsolatedAsyncioTestCase):
  """Validates multi-modal coercion and InputEvent serialization inside LocalConnection.send()."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_send_flat_string_populates_user_input(self):
    """Verifies that a standard string prompt maps to the user_input proto field."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.conn.send("Standard text prompt")

    sent_data = await harness.wait_for_response()
    self.assertIn("userInput", sent_data)
    parts = sent_data["userInput"]["parts"]
    self.assertEqual(len(parts), 1)
    self.assertEqual(parts[0]["text"], "Standard text prompt")

  async def test_send_none_prompt_populates_blank_string(self):
    """Verifies that passing a prompt of None maps to a blank userInput text part."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.conn.send(None)

    sent_data = await harness.wait_for_response()

    self.assertIn("userInput", sent_data)
    parts = sent_data["userInput"]["parts"]
    self.assertEqual(len(parts), 1)
    self.assertEqual(parts[0].get("text", ""), "")

  async def test_send_single_media_content_populates_user_input(self):
    """Verifies that a single rich Content primitive maps to the user_input parts list."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    image_content = types.Image(
        mime_type="image/png",
        data=b"fake_png",
        description="logo image",
    )
    await harness.conn.send(image_content)

    sent_data = await harness.wait_for_response()

    self.assertIn("userInput", sent_data)

    parts = sent_data["userInput"]["parts"]
    self.assertEqual(len(parts), 1)
    self.assertIn("media", parts[0])
    media = parts[0]["media"]
    self.assertEqual(media["mimeType"], "image/png")
    self.assertEqual(media["description"], "logo image")
    # Protobuf JSON automatically base64-encodes binary bytes
    self.assertEqual(media["data"], "ZmFrZV9wbmc=")  # b"fake_png"

  async def test_send_mixed_list_populates_multiple_complex_content(self):
    """Verifies that a list containing both strings and rich Content primitives compiles correctly to spec."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    mixed_prompt = [
        "Context text instruction.",
        types.Document(mime_type="application/pdf", data=b"fake_pdf"),
    ]
    await harness.conn.send(mixed_prompt)

    sent_data = await harness.wait_for_response()

    self.assertIn("userInput", sent_data)

    parts = sent_data["userInput"]["parts"]
    self.assertEqual(len(parts), 2)

    self.assertEqual(parts[0]["text"], "Context text instruction.")

    self.assertEqual(parts[1]["media"]["mimeType"], "application/pdf")
    self.assertEqual(parts[1]["media"]["data"], "ZmFrZV9wZGY=")  # b"fake_pdf"

  async def test_send_slash_command_populates_user_input(self):
    """Verifies that a SlashCommand primitive maps to user_input slash_command field."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    slash_command = types.SlashCommand(
        name=types.BuiltinSlashCommandName.PLAN,
    )
    await harness.conn.send(slash_command)

    sent_data = await harness.wait_for_response()

    self.assertIn("userInput", sent_data)

    parts = sent_data["userInput"]["parts"]
    self.assertEqual(len(parts), 1)
    self.assertIn("slashCommand", parts[0])
    sc = parts[0]["slashCommand"]
    self.assertEqual(sc["name"], "plan")

  async def test_concurrent_receive_steps_raises(self):
    """Verifies that a second concurrent receive_steps() call raises RuntimeError.

    The connection sets _is_receiving on entry and clears it on exit.
    A second caller must fail immediately with a clear message rather than
    silently racing on the shared step queue.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    # Put the connection into a non-idle state so receive_steps blocks
    # waiting for steps.
    harness.conn._is_idle.clear()

    first_started = asyncio.Event()

    async def _first_receiver():
      first_started.set()
      async for _ in harness.conn.receive_steps():
        pass  # Will block on the queue until idle or close sentinel.

    task = asyncio.create_task(_first_receiver())
    await first_started.wait()
    # Give the first receiver a moment to enter the iterator body.
    await asyncio.sleep(0.05)

    with self.assertRaises(RuntimeError) as ctx:
      async for _ in harness.conn.receive_steps():
        pass

    self.assertIn(
        "Concurrent receive_steps() calls are not supported", str(ctx.exception)
    )

    # Clean up: signal idle so the first receiver can exit.
    harness.conn._is_idle.set()
    await harness.conn._step_queue.put(local_connection.IDLE_SENTINEL)
    task.cancel()
    try:
      await task
    except asyncio.CancelledError:
      pass

  async def test_trigger_notification_succeeds_while_busy(self):
    """Verifies send_trigger_notification() and send() work during an active turn.

    There is no connection-level guard on sending while a receive is in
    progress. Callers (e.g. scheduled triggers) must be able to inject
    new prompts even when steps are still being consumed.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    # Start a turn so the connection is non-idle.
    await harness.conn.send("initial prompt")
    initial_msg = await harness.wait_for_response()
    self.assertEqual(
        initial_msg.get("userInput"), {"parts": [{"text": "initial prompt"}]}
    )

    # send_trigger_notification should succeed even though we are mid-turn.
    await harness.conn.send_trigger_notification("trigger content")
    trigger_msg = await harness.wait_for_response()
    self.assertIn("automatedTrigger", trigger_msg)
    self.assertEqual(trigger_msg["automatedTrigger"], "trigger content")

    # A regular send() should also succeed (no send-side guard).
    await harness.conn.send("follow-up prompt")
    followup_msg = await harness.wait_for_response()
    self.assertEqual(
        followup_msg.get("userInput"),
        {"parts": [{"text": "follow-up prompt"}]},
    )


class LocalAgentConfigTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)

  def test_create_strategy(self):
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test instructions",
        model="gemini-2.5-pro",
    )

    mock_tool_runner = mock.create_autospec(
        tool_runner.ToolRunner, instance=True
    )
    mock_hook_runner = mock.create_autospec(
        hook_runner.HookRunner, instance=True
    )

    strategy = config.create_strategy(
        tool_runner=mock_tool_runner,
        hook_runner=mock_hook_runner,
    )

    self.assertIsInstance(strategy, local_connection.LocalConnectionStrategy)
    self.assertIsNotNone(strategy._models)
    text_models = [
        m for m in strategy._models if types.ModelType.TEXT in m.types
    ]
    self.assertLen(text_models, 1)
    self.assertEqual(text_models[0].name, "gemini-2.5-pro")

  def test_create_strategy_passes_env(self):
    config = local_connection_config.LocalAgentConfig(
        env={"CUSTOM_KEY": "CUSTOM_VAL"},
    )
    mock_tool_runner = mock.create_autospec(
        tool_runner.ToolRunner, instance=True
    )
    mock_hook_runner = mock.create_autospec(
        hook_runner.HookRunner, instance=True
    )
    strategy = config.create_strategy(
        tool_runner=mock_tool_runner,
        hook_runner=mock_hook_runner,
    )
    self.assertEqual(strategy._env, {"CUSTOM_KEY": "CUSTOM_VAL"})

  def test_merge_models_only_defaults(self):
    config = local_connection_config.LocalAgentConfig()
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, models_lib.DEFAULT_MODEL)
    self.assertEqual(config.models[0].types, [types.ModelType.TEXT])
    self.assertEqual(
        config.models[1].name,
        local_connection_config.DEFAULT_IMAGE_GENERATION_MODEL,
    )
    self.assertEqual(config.models[1].types, [types.ModelType.IMAGE])

  def test_merge_models_shorthand_only(self):
    config = local_connection_config.LocalAgentConfig(model="custom-text-model")
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "custom-text-model")
    self.assertEqual(config.models[0].types, [types.ModelType.TEXT])
    self.assertEqual(
        config.models[1].name,
        local_connection_config.DEFAULT_IMAGE_GENERATION_MODEL,
    )
    self.assertEqual(config.models[1].types, [types.ModelType.IMAGE])

  def test_merge_models_explicit_only(self):
    custom_image = types.ModelTarget(
        name="custom-image-model", types=[types.ModelType.IMAGE]
    )
    config = local_connection_config.LocalAgentConfig(models=[custom_image])
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "custom-image-model")
    self.assertEqual(config.models[0].types, [types.ModelType.IMAGE])
    self.assertEqual(config.models[1].name, models_lib.DEFAULT_MODEL)
    self.assertEqual(config.models[1].types, [types.ModelType.TEXT])

  def test_merge_models_explicit_and_shorthand(self):
    custom_image = types.ModelTarget(
        name="custom-image-model", types=[types.ModelType.IMAGE]
    )
    config = local_connection_config.LocalAgentConfig(
        model="custom-text-model", models=[custom_image]
    )
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "custom-image-model")
    self.assertEqual(config.models[0].types, [types.ModelType.IMAGE])
    self.assertEqual(config.models[1].name, "custom-text-model")
    self.assertEqual(config.models[1].types, [types.ModelType.TEXT])

  def test_constructor_parameters_fully_typed(self):
    """Verifies all subclass fields are accepted by the constructor under pytype."""
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(enable_subagents=True),
        tools=[],
        policies=[],
        hooks=[],
        triggers=[],
        mcp_servers=[],
        workspaces=["/tmp/ws"],
        conversation_id="12345678901234567890123456789012",
        save_dir="/tmp/save",
        app_data_dir="/tmp/app",
        response_schema="{}",
        skills_paths=["/tmp/skills"],
        model="gemini-2.5-pro",
        api_key="fake_api_key",
        vertex=True,
        project="my_project",
        location="us-central1",
    )
    self.assertEqual(config.model, "gemini-2.5-pro")
    self.assertEqual(config.api_key, "fake_api_key")
    self.assertTrue(config.vertex)
    self.assertEqual(config.project, "my_project")
    self.assertEqual(config.location, "us-central1")
    self.assertEqual(config.conversation_id, "12345678901234567890123456789012")

  def test_safe_defaults(self):
    """LocalAgentConfig defaults to confirm_run_command() — deny run_command."""
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        workspaces=[],
    )
    self.assertIsNone(config.capabilities.enabled_tools)
    self.assertIsNone(config.capabilities.disabled_tools)
    # confirm_run_command() produces 2 policies: deny(run_command) + allow(*)
    self.assertLen(config.policies, 2)
    deny_policy = config.policies[0]
    self.assertEqual(deny_policy.tool, "run_command")
    self.assertEqual(deny_policy.decision, policy.Decision.DENY)
    self.assertEqual(deny_policy.name, "confirm_run_command")
    allow_policy = config.policies[1]
    self.assertEqual(allow_policy.tool, "*")
    self.assertEqual(allow_policy.decision, policy.Decision.APPROVE)

  def test_lightweight_method(self):
    config = local_connection_config.LocalAgentConfig(
        model="gemini-3.8-flash",
    ).lightweight()
    self.assertIsInstance(config, local_connection_config.LocalAgentConfig)
    self.assertEqual(config.model, "gemini-3.8-flash")
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )
    self.assertEqual(config.capabilities.compaction_threshold, 65536)
    self.assertFalse(config.capabilities.enable_subagents)

    # Verify HarnessConfig proto serialization with agent_behavior
    strategy = config.create_strategy(tool_runner=None, hook_runner=None)
    harness_config = strategy._build_harness_config()
    self.assertEqual(
        harness_config.agent_behavior,
        localharness_pb2.AGENT_BEHAVIOR_MINIMAL,
    )
    self.assertEqual(harness_config.compaction_threshold, 65536)
    self.assertFalse(harness_config.harness_side_tools.subagents.enabled)
    self.assertTrue(harness_config.harness_side_tools.run_command.enabled)
    self.assertTrue(harness_config.harness_side_tools.view_file.enabled)
    self.assertTrue(harness_config.harness_side_tools.write_to_file.enabled)
    self.assertTrue(harness_config.harness_side_tools.file_edit.enabled)
    self.assertTrue(harness_config.harness_side_tools.list_dir.enabled)
    self.assertTrue(harness_config.harness_side_tools.grep_search.enabled)
    self.assertFalse(harness_config.harness_side_tools.find.enabled)
    self.assertFalse(harness_config.harness_side_tools.user_questions.enabled)

  def test_lightweight_method_with_overrides(self):
    config = local_connection_config.LocalAgentConfig(
        model="gemini-2.5-flash-lite",
        capabilities=types.CapabilitiesConfig(
            compaction_threshold=8000,
        ),
    ).lightweight()
    self.assertIsInstance(config, local_connection_config.LocalAgentConfig)
    self.assertEqual(config.model, "gemini-2.5-flash-lite")
    self.assertEqual(config.capabilities.compaction_threshold, 8000)
    self.assertFalse(config.capabilities.enable_subagents)
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.MINIMAL
    )
    self.assertEqual(
        config.capabilities.enabled_tools, types.BuiltinTools.minimal()
    )

  def test_safe_defaults_with_default_workspace(self):
    """LocalAgentConfig defaults to CWD workspace when not specified."""
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
    )
    self.assertEqual(config.workspaces, [os.getcwd()])
    # workspace containment is enforced natively by localharness;
    # config.policies contains only the 2 confirm_run_command policies.
    self.assertLen(config.policies, 2)
    self.assertEqual(config.policies[0].tool, "run_command")
    self.assertEqual(config.policies[1].tool, "*")

  def test_explicit_allow_all_overrides_default(self):
    """Explicit allow_all() replaces the confirm_run_command default."""
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        policies=[policy.allow_all()],
        workspaces=[],
    )
    self.assertLen(config.policies, 1)
    self.assertEqual(config.policies[0].tool, "*")
    self.assertEqual(config.policies[0].decision, policy.Decision.APPROVE)

  def test_create_strategy_app_data_dir(self):
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test instructions",
        app_data_dir="/foo/bar",
    )

    mock_tool_runner = mock.create_autospec(
        tool_runner.ToolRunner, instance=True
    )
    mock_hook_runner = mock.create_autospec(
        hook_runner.HookRunner, instance=True
    )

    strategy = config.create_strategy(
        tool_runner=mock_tool_runner,
        hook_runner=mock_hook_runner,
    )

    self.assertIsInstance(strategy, local_connection.LocalConnectionStrategy)
    self.assertEqual(strategy._app_data_dir, "/foo/bar")

  def test_app_data_dir_relative_path_raises(self):
    with self.assertRaises(pydantic.ValidationError):
      local_connection_config.LocalAgentConfig(
          system_instructions="test",
          app_data_dir="relative/path",
      )

  def test_conversation_id_validation(self):
    # Valid ID (32 chars, alphanumeric)
    local_connection_config.LocalAgentConfig(
        system_instructions="test",
        conversation_id="12345678901234567890123456789012",
    )

    # Valid ID (36 chars, UUID format with hyphens)
    local_connection_config.LocalAgentConfig(
        system_instructions="test",
        conversation_id="12345678-1234-1234-1234-123456789012",
    )

    # Invalid ID (too short)
    with self.assertRaises(pydantic.ValidationError) as ctx:
      local_connection_config.LocalAgentConfig(
          system_instructions="test",
          conversation_id="too-short",
      )
    self.assertIn("must be at least 32 characters long", str(ctx.exception))

    # Invalid ID (invalid characters)
    with self.assertRaises(pydantic.ValidationError) as ctx:
      local_connection_config.LocalAgentConfig(
          system_instructions="test",
          conversation_id="invalid_char_because_of_underscores_123",
      )
    self.assertIn("must match [a-zA-Z0-9-]", str(ctx.exception))

  def test_local_agent_config_validates_allowed_subagents_success(self):
    sub = types.SubagentConfig(
        name="researcher",
        description="researcher",
        system_instructions="research",
    )
    config = local_connection_config.LocalAgentConfig(
        subagents=[sub],
        capabilities=types.CapabilitiesConfig(
            allowed_subagents=["researcher"],
        ),
    )
    self.assertEqual(config.capabilities.allowed_subagents, ["researcher"])

  def test_local_agent_config_validates_allowed_subagents_unknown_raises(self):
    sub = types.SubagentConfig(
        name="researcher",
        description="researcher",
        system_instructions="research",
    )
    with self.assertRaisesRegex(
        pydantic.ValidationError, "Unknown subagent name.*non_existent"
    ):
      local_connection_config.LocalAgentConfig(
          subagents=[sub],
          capabilities=types.CapabilitiesConfig(
              allowed_subagents=["non_existent"],
          ),
      )

  def test_local_agent_config_validates_subagent_allowed_subagents_unknown_raises(
      self,
  ):
    sub = types.SubagentConfig(
        name="researcher",
        description="researcher",
        capabilities=types.SubagentCapabilities(
            enabled_tools=[types.BuiltinTools.START_SUBAGENT],
            allowed_subagents=["ghost_agent"],
        ),
    )
    with self.assertRaisesRegex(
        pydantic.ValidationError, "Unknown subagent name.*ghost_agent"
    ):
      local_connection_config.LocalAgentConfig(
          subagents=[sub],
      )

  def test_create_strategy_with_mcp_servers(self):
    stdio_cfg = types.McpStdioServer(
        name="my-stdio",
        command="npx",
        args=["math"],
        env={"FOO": "bar"},
        enabled_tools=["add", "sub"],
    )
    sse_cfg = types.McpStreamableHttpServer(
        name="my-sse",
        url="https://sse.example.com",
    )
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        mcp_servers=[stdio_cfg, sse_cfg],
        api_key="fake",
    )

    mock_tool_runner = mock.create_autospec(
        tool_runner.ToolRunner, instance=True
    )
    mock_hook_runner = mock.create_autospec(
        hook_runner.HookRunner, instance=True
    )

    strategy = config.create_strategy(
        tool_runner=mock_tool_runner,
        hook_runner=mock_hook_runner,
    )

    harness_pb = strategy._build_harness_config()

    self.assertLen(harness_pb.mcp_servers, 2)

    stdio_pb = harness_pb.mcp_servers[0]
    self.assertEqual(stdio_pb.name, "my-stdio")
    self.assertEqual(stdio_pb.enabled_tools, ["add", "sub"])
    self.assertEqual(stdio_pb.stdio.command, "npx")
    self.assertEqual(stdio_pb.stdio.args, ["math"])
    self.assertEqual(dict(stdio_pb.stdio.env), {"FOO": "bar"})

    sse_pb = harness_pb.mcp_servers[1]
    self.assertEqual(sse_pb.name, "my-sse")
    self.assertEqual(sse_pb.http.url, "https://sse.example.com")


class LocalConnectionBuiltinToolHooksTest(unittest.IsolatedAsyncioTestCase):
  """Tests for built-in tool STATE_DONE/STATE_ERROR cleanup.

  Built-in tools (run_command, list_directory, etc.) execute inside the Go
  harness. PostToolCallHook is now dispatched by Go via CallHookRequest (tested
  in hook_router_test.py). The Python SDK tracks approved tool calls via
  _pending_builtin_tool_calls to handle STATE_ERROR (OnToolErrorHook, to be
  migrated in a follow-up CL).

  These tests verify that STATE_DONE properly cleans up the pending tracking
  without Python-side hook dispatch.
  """

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock(spec=subprocess.Popen)
    self.tool_runner = tool_runner.ToolRunner()

  def _make_harness(self, hr):
    """Creates a TestLocalHarness with the given HookRunner."""
    return test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

  async def _confirm_and_complete(self, harness, confirm_event, done_event):
    """Sends a confirmation request and then a completion event.

    Args:
      harness: The TestLocalHarness instance.
      confirm_event: The OutputEvent with STATE_WAITING_FOR_USER.
      done_event: The OutputEvent with STATE_DONE or STATE_ERROR.

    Returns:
      The confirmation response dict from the SDK.
    """
    await harness.send_event(confirm_event)
    sent_data = await harness.wait_for_response()
    await harness.send_event(done_event)
    return sent_data

  def _make_confirm_event(self, step_index, traj_id, **action_kwargs):
    """Builds a STATE_WAITING_FOR_USER OutputEvent with an action field."""
    return localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=step_index,
            trajectory_id=traj_id,
            cascade_id=traj_id,
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            **action_kwargs,
        )
    )

  def _make_done_event(self, step_index, traj_id, **action_kwargs):
    """Builds a STATE_DONE OutputEvent with an action field."""
    return localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=step_index,
            trajectory_id=traj_id,
            cascade_id=traj_id,
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            **action_kwargs,
        )
    )

  # ---- Guard tests ----

  async def test_denied_builtin_not_tracked(self):
    """Verifies denied builtin tools don't trigger post-tool dispatch.

    What: PostToolCallHook does NOT fire for denied built-in tool calls.
    Why: If a Decide hook denies a builtin tool, the harness rejects it and
         there is no execution to observe.
    How: Deny via Decide hook, send a STATE_DONE for the same step, and
         verify PostToolCallHook was not called.
    """
    hook_fired = asyncio.Event()

    class DenyHook(hooks_base.PreToolCallDecideHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        return hooks_base.HookResult(allow=False, message="Denied")

    class PostHook(hooks_base.PostToolCallHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        hook_fired.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(DenyHook())
    hr.register_hook(PostHook())
    harness = self._make_harness(hr)

    # After migration, _handle_tool_confirmation_request always auto-accepts.
    # Pre-tool deny logic is now in HookRouter._handle_pre_tool.
    # This test verifies that the confirmation path auto-accepts even with
    # a deny hook registered.
    await harness.send_event(
        self._make_confirm_event(
            0,
            "traj_deny",
            view_file=localharness_pb2.ActionViewFile(file_path="/foo"),
        )
    )
    sent_data = await harness.wait_for_response()
    self.assertTrue(sent_data["toolConfirmation"]["accepted"])

  async def test_no_spurious_hook_for_non_builtin_step(self):
    """Verifies post-tool hooks don't fire for normal model response steps.

    What: PostToolCallHook does NOT fire for STATE_DONE model response steps.
    Why: Only steps that were tracked via ToolConfirmation should trigger
         PostToolCallHook. A model response step that happens to be STATE_DONE
         must not be confused with a completed builtin tool.
    How: Send a model response step (no prior confirmation) and verify
         PostToolCallHook was not called.
    """
    hook_fired = asyncio.Event()

    class PostHook(hooks_base.PostToolCallHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        hook_fired.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(PostHook())
    harness = self._make_harness(hr)

    # A normal model step (not a builtin tool) that is DONE.
    await harness.send_event(
        localharness_pb2.OutputEvent(
            step_update=localharness_pb2.StepUpdate(
                cascade_id="traj",
                trajectory_id="traj",
                step_index=5,
                text="Final model response",
                state=localharness_pb2.StepUpdate.STATE_DONE,
                source=localharness_pb2.StepUpdate.SOURCE_MODEL,
                target=localharness_pb2.StepUpdate.TARGET_USER,
            )
        )
    )
    await asyncio.sleep(0.1)
    self.assertFalse(hook_fired.is_set())


class LocalConnectionExceptionSafetyTest(unittest.IsolatedAsyncioTestCase):
  """Tests verifying that handler exceptions don't deadlock the harness.

  Each background handler (_handle_question_request,
  _handle_tool_confirmation_request, _handle_tool_call) must catch
  exceptions and send an informative error response rather than dying
  silently and leaving the Go harness blocked.
  """

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock(spec=subprocess.Popen)
    self.tool_runner = tool_runner.ToolRunner()

  def _make_harness(self, hr=None):
    return test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

  async def test_question_handler_crash_sends_error(self):
    """Verifies a crashing interaction hook sends the error message.

    When the on_interaction hook raises, the handler must still respond
    to the harness to prevent deadlock. The error is sent as a single
    freeform_response answer so the model sees what happened.
    """
    hr = hook_runner.HookRunner()

    @hooks_base.on_interaction
    async def crashing_hook(data):
      _ = data
      raise RuntimeError("Intentional interaction hook crash")

    hr.register_hook(crashing_hook)
    harness = self._make_harness(hr)

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            questions_request=localharness_pb2.UserQuestionsRequest(
                questions=[
                    localharness_pb2.UserQuestion(
                        multiple_choice=localharness_pb2.MultipleChoice(
                            question="Do you agree?",
                            choices=["Yes", "No"],
                        )
                    )
                ]
            ),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("questionResponse", sent_data)
    resp = sent_data["questionResponse"]["response"]
    answers = resp["answers"]
    # Single answer with the error in freeform_response.
    self.assertEqual(len(answers), 1)
    freeform = answers[0]["multipleChoiceAnswer"]["freeformResponse"]
    self.assertIn("SDK error", freeform)
    self.assertIn("Intentional interaction hook crash", freeform)


class LocalConnectionSerializationTest(unittest.IsolatedAsyncioTestCase):
  """Tests verifying Pydantic-based normalization in _tool_result_to_dict.

  The SDK uses pydantic.TypeAdapter(Any) to normalize tool outputs
  into JSON-safe primitives before json.dumps(). This prevents
  serialization errors (the root cause of the deadlock bug) when tools
  return complex Python types like sets, datetimes, or bytes.
  """

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock(spec=subprocess.Popen)
    self.tool_runner = tool_runner.ToolRunner()

  def _make_harness(self):
    return test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

  async def test_normalizes_set_to_list(self):
    """Verifies _tool_result_to_dict normalizes sets into JSON lists.

    This is the exact type that triggered the original deadlock: a tool
    returning a set caused json.dumps to raise TypeError, killing the
    background task and leaving the harness waiting forever.
    """
    conn = self._make_harness().conn
    tr = types.ToolResult(id="1", name="t", result={"tags": {"python", "sdk"}})
    res_dict = conn._tool_result_to_dict(tr)
    self.assertIsInstance(res_dict["tags"], list)
    self.assertCountEqual(res_dict["tags"], ["python", "sdk"])

  async def test_normalizes_datetime_to_iso_string(self):
    """Verifies _tool_result_to_dict normalizes datetimes into ISO strings."""
    conn = self._make_harness().conn
    dt = datetime.datetime(2026, 5, 15, 2, 30, 0)
    tr = types.ToolResult(id="1", name="t", result={"time": dt})
    self.assertEqual(
        conn._tool_result_to_dict(tr)["time"], "2026-05-15T02:30:00"
    )

  async def test_normalizes_bytes_to_string(self):
    """Verifies _tool_result_to_dict normalizes bytes into UTF-8 strings."""
    conn = self._make_harness().conn
    tr = types.ToolResult(id="1", name="t", result={"data": b"hello"})
    self.assertEqual(conn._tool_result_to_dict(tr)["data"], "hello")

  async def test_preserves_pydantic_custom_serializer(self):
    """Verifies _tool_result_to_dict respects custom @field_serializer.

    When a tool returns a Pydantic model with a custom serializer (e.g.
    to mask secrets), model_dump(mode="json") must be used instead of
    model_dump() to ensure the serializer runs.
    """
    conn = self._make_harness().conn

    class CustomModel(pydantic.BaseModel):
      secret: str

      @pydantic.field_serializer("secret")
      def mask_secret(
          self, secret: str, info: pydantic.FieldSerializationInfo
      ) -> str:
        del secret, info
        return "xxxx"

    tr = types.ToolResult(
        id="call_1",
        name="test_tool",
        result=CustomModel(secret="my_super_secret_key"),
    )

    res_dict = conn._tool_result_to_dict(tr)
    self.assertEqual(res_dict["secret"], "xxxx")


class LocalConnectionSubagentsTest(unittest.IsolatedAsyncioTestCase):
  """Tests verifying that static subagent configs are built into HarnessConfig."""

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)
    self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
    self.workspace = pathlib.Path(self.temp_dir) / "workspace"
    self.workspace.mkdir()

  def test_builds_subagents_proto_correctly(self):
    def my_custom_tool():
      """A test tool."""

    def another_one():
      """Another test tool."""

    subagent = types.SubagentConfig(
        name="test_helper",
        description="A helpful subagent for testing",
        system_instructions="Always say hello.",
        capabilities=types.SubagentCapabilities(
            enabled_tools=[
                types.BuiltinTools.EDIT_FILE,
            ],
        ),
        # Test mixing callable tools and string tools
        tools=[my_custom_tool, "another_one"],
    )

    tr = tool_runner.ToolRunner(tools=[my_custom_tool, another_one])

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
        tool_runner=tr,
    )

    harness_config = strategy._build_harness_config()

    self.assertEqual(len(harness_config.custom_subagents), 1)
    custom_agent = harness_config.custom_subagents[0]
    self.assertEqual(custom_agent.name, "test_helper")
    self.assertEqual(custom_agent.description, "A helpful subagent for testing")
    self.assertTrue(custom_agent.harness_side_tools.file_edit.enabled)
    self.assertFalse(custom_agent.harness_side_tools.view_file.enabled)
    self.assertFalse(custom_agent.harness_side_tools.subagents.enabled)
    self.assertEqual(
        [t.name for t in custom_agent.tools],
        ["my_custom_tool", "another_one"],
    )
    sections = custom_agent.system_instructions.appended.appended_sections
    self.assertEqual(sections[0].title, "user_system_instructions")
    self.assertEqual(sections[0].content, "Always say hello.")

  def test_builds_subagents_proto_with_sections(self):
    subagent = types.SubagentConfig(
        name="test_helper",
        description="A helpful subagent for testing",
        system_instructions=types.TemplatedSystemInstructions(
            sections=[
                types.SystemInstructionSection(
                    title="Identity", content="You are a helper agent."
                ),
                types.SystemInstructionSection(
                    title="Guidelines", content="Keep responses short."
                ),
            ]
        ),
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    harness_config = strategy._build_harness_config()

    self.assertEqual(len(harness_config.custom_subagents), 1)
    custom_agent = harness_config.custom_subagents[0]
    self.assertEqual(custom_agent.name, "test_helper")
    sections = custom_agent.system_instructions.appended.appended_sections
    self.assertEqual(len(sections), 2)
    self.assertEqual(sections[0].title, "Identity")
    self.assertEqual(sections[0].content, "You are a helper agent.")
    self.assertEqual(sections[1].title, "Guidelines")
    self.assertEqual(sections[1].content, "Keep responses short.")

  def test_builds_subagents_proto_with_custom_system_instructions(self):
    subagent = types.SubagentConfig(
        name="custom_helper",
        description="A subagent with custom instructions",
        system_instructions=types.CustomSystemInstructions(
            text="Fully custom subagent instructions."
        ),
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    harness_config = strategy._build_harness_config()

    self.assertEqual(len(harness_config.custom_subagents), 1)
    custom_agent = harness_config.custom_subagents[0]
    self.assertEqual(custom_agent.name, "custom_helper")
    parts = custom_agent.system_instructions.custom.part
    self.assertEqual(len(parts), 1)
    self.assertEqual(parts[0].text, "Fully custom subagent instructions.")

  def test_builds_subagents_proto_with_templated_system_instructions(self):
    subagent = types.SubagentConfig(
        name="templated_helper",
        description="A subagent with templated instructions",
        system_instructions=types.TemplatedSystemInstructions(
            identity="Subagent identity",
            sections=[
                types.SystemInstructionSection(
                    title="Section1", content="Content1"
                )
            ],
        ),
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    harness_config = strategy._build_harness_config()

    self.assertEqual(len(harness_config.custom_subagents), 1)
    custom_agent = harness_config.custom_subagents[0]
    self.assertEqual(custom_agent.name, "templated_helper")
    appended = custom_agent.system_instructions.appended
    self.assertEqual(appended.custom_identity, "Subagent identity")
    self.assertEqual(len(appended.appended_sections), 1)
    self.assertEqual(appended.appended_sections[0].title, "Section1")
    self.assertEqual(appended.appended_sections[0].content, "Content1")

  def test_subagent_exclusive_tool_scoping(self):
    def main_tool():
      """Main agent tool."""

    def sub_tool():
      """Subagent exclusive tool."""

    subagent = types.SubagentConfig(
        name="test_helper",
        description="A helpful subagent",
        tools=[sub_tool],
    )

    tr = tool_runner.ToolRunner(tools=[main_tool, sub_tool])

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
        tool_runner=tr,
        tools=[main_tool],
    )

    harness_config = strategy._build_harness_config()

    # Main agent tools should only have main_tool.
    self.assertEqual([t.name for t in harness_config.tools], ["main_tool"])
    # Subagent tools should only have sub_tool.
    self.assertEqual(len(harness_config.custom_subagents), 1)
    self.assertEqual(
        [t.name for t in harness_config.custom_subagents[0].tools],
        ["sub_tool"],
    )

  def test_subagent_exclusive_callable_tool_resolved_automatically(self):
    def sub_only_tool():
      """Only on subagent without tool_runner."""

    subagent = types.SubagentConfig(
        name="test_helper",
        description="A helpful subagent",
        tools=[sub_only_tool],
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
        tools=[],
    )

    harness_config = strategy._build_harness_config()

    self.assertEqual(len(harness_config.tools), 0)
    self.assertEqual(len(harness_config.custom_subagents), 1)
    self.assertEqual(
        [t.name for t in harness_config.custom_subagents[0].tools],
        ["sub_only_tool"],
    )

  def test_subagent_callable_functor_without_name(self):
    class FunctorTool:

      def __call__(self, x: int) -> int:
        """A functor tool."""
        return x * 2

    functor = FunctorTool()
    subagent = types.SubagentConfig(
        name="functor_helper",
        description="Helper with functor",
        tools=[functor],
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
        tools=[],
    )

    harness_config = strategy._build_harness_config()
    self.assertEqual(len(harness_config.custom_subagents), 1)
    self.assertEqual(
        [t.name for t in harness_config.custom_subagents[0].tools],
        ["FunctorTool"],
    )

  def test_build_harness_config_tools_none_fallback(self):
    def root_tool():
      """Root tool."""

    def sub_tool():
      """Sub tool."""

    subagent = types.SubagentConfig(
        name="sub",
        description="sub",
        tools=[sub_tool],
    )
    tr = tool_runner.ToolRunner(tools=[root_tool, sub_tool])
    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
        tool_runner=tr,
        # tools not specified (None)
    )

    harness_config = strategy._build_harness_config()
    self.assertEqual([t.name for t in harness_config.tools], ["root_tool"])
    self.assertEqual(
        [t.name for t in harness_config.custom_subagents[0].tools],
        ["sub_tool"],
    )

  def test_subagent_string_named_tools_builds_tool_protos(self):
    subagent = types.SubagentConfig(
        name="string_tool_helper",
        description="Helper with string tools",
        tools=["view_file", "code_search"],
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    harness_config = strategy._build_harness_config()
    self.assertEqual(len(harness_config.custom_subagents), 1)
    self.assertEqual(
        [t.name for t in harness_config.custom_subagents[0].tools],
        ["view_file", "code_search"],
    )

  def test_subagent_invalid_tool_type_raises_value_error(self):
    subagent = types.SubagentConfig.model_construct(
        name="invalid_helper",
        description="Helper with invalid tool",
        tools=[12345],  # pytype: disable=wrong-arg-types
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    with self.assertRaisesRegex(
        ValueError,
        "Invalid tool type in subagent 'invalid_helper' tools list: 12345",
    ):
      strategy._build_harness_config()

  def test_subagent_with_start_subagent_enables_nested_delegation(self):
    """Verifies subagents with START_SUBAGENT get subagents.enabled=True."""
    subagent = types.SubagentConfig(
        name="nested_helper",
        description="A subagent that can spawn subagents",
        system_instructions="Spawn subagents.",
        capabilities=types.SubagentCapabilities(
            enabled_tools=[types.BuiltinTools.START_SUBAGENT],
        ),
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    harness_config = strategy._build_harness_config()

    self.assertEqual(len(harness_config.custom_subagents), 1)
    custom_agent = harness_config.custom_subagents[0]
    self.assertTrue(custom_agent.harness_side_tools.subagents.enabled)
    self.assertFalse(custom_agent.harness_side_tools.file_edit.enabled)

  def test_local_agent_config_subagents_none_initializes(self):
    config = local_connection_config.LocalAgentConfig(subagents=None)
    self.assertEqual(config.subagents, [])

  def test_local_agent_config_kwargs_none_filtered(self):
    config = local_connection_config.LocalAgentConfig(
        **{"subagents": None, "capabilities": None, "conversation_id": None}
    )
    self.assertEqual(config.subagents, [])
    self.assertIsInstance(config.capabilities, types.CapabilitiesConfig)
    self.assertIsNone(config.conversation_id)


if __name__ == "__main__":
  absltest.main()
