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

"""Tests for event_processor that translates wire events to SDK events."""

import asyncio
import json
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from google.protobuf import json_format
import pydantic

from google.antigravity.proto import content_pb2
from google.antigravity.proto import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local import event_processor
from google.antigravity.connections.local import types as local_types
from google.antigravity.hooks import policy


MAIN_TRAJECTORY_ID = "cbb3a5135a32671ae8152a25a857c4bc"
SUBAGENT_TRAJECTORY_ID = "9121f3e9937e263b74a4a43ff6fb0117"


class EventProcessorHelperTest(absltest.TestCase):
  """Tests for standalone helper functions in event_processor."""

  def test_normalize_wire_path_file_uri(self):
    self.assertEqual(
        event_processor.normalize_wire_path("file:///dev/shm/workspace/foo.py"),
        "/dev/shm/workspace/foo.py",
    )

  def test_normalize_wire_path_cns_uri(self):
    self.assertEqual(
        event_processor.normalize_wire_path(
            "cns://el-d/home/user/workspace/kittens.md"
        ),
        "/cns/el-d/home/user/workspace/kittens.md",
    )

  def test_normalize_wire_path_plain_path(self):
    self.assertEqual(
        event_processor.normalize_wire_path("/tmp/clean-path"),
        "/tmp/clean-path",
    )

  def test_make_step_id_with_trajectory(self):
    self.assertEqual(event_processor._make_step_id("traj_1", 5), "traj_1:5")

  def test_make_step_id_without_trajectory(self):
    self.assertEqual(event_processor._make_step_id("", 5), "5")

  def test_parse_usage_metadata_full(self):
    pb = localharness_pb2.UsageMetadata(
        prompt_token_count=100,
        cached_content_token_count=50,
        candidates_token_count=75,
        thoughts_token_count=25,
        total_token_count=250,
        service_tier="priority",
    )
    meta = event_processor.parse_usage_metadata(pb)
    self.assertEqual(meta.prompt_token_count, 100)
    self.assertEqual(meta.cached_content_token_count, 50)
    self.assertEqual(meta.candidates_token_count, 75)
    self.assertEqual(meta.thoughts_token_count, 25)
    self.assertEqual(meta.total_token_count, 250)
    self.assertEqual(meta.service_tier, types.ServiceTier.PRIORITY)

  def test_parse_usage_metadata_empty(self):
    pb = localharness_pb2.UsageMetadata()
    meta = event_processor.parse_usage_metadata(pb)
    self.assertIsNone(meta.prompt_token_count)
    self.assertIsNone(meta.cached_content_token_count)
    self.assertIsNone(meta.candidates_token_count)
    self.assertIsNone(meta.thoughts_token_count)
    self.assertIsNone(meta.total_token_count)
    self.assertIsNone(meta.service_tier)

  def test_parse_stop_reason(self):
    self.assertEqual(
        event_processor._parse_stop_reason(  # pylint: disable=protected-access
            localharness_pb2.TrajectoryStateUpdate.STOP_REASON_MAX_MODEL_CALLS_EXCEEDED
        ),
        types.StopReason.MAX_MODEL_CALLS_EXCEEDED,
    )
    self.assertEqual(
        event_processor._parse_stop_reason(  # pylint: disable=protected-access
            localharness_pb2.TrajectoryStateUpdate.STOP_REASON_MAX_TOOL_CALLS_EXCEEDED
        ),
        types.StopReason.MAX_TOOL_CALLS_EXCEEDED,
    )
    self.assertEqual(
        event_processor._parse_stop_reason(  # pylint: disable=protected-access
            localharness_pb2.TrajectoryStateUpdate.STOP_REASON_MAX_INPUT_TOKENS_EXCEEDED
        ),
        types.StopReason.MAX_INPUT_TOKENS_EXCEEDED,
    )
    self.assertEqual(
        event_processor._parse_stop_reason(  # pylint: disable=protected-access
            localharness_pb2.TrajectoryStateUpdate.STOP_REASON_MAX_OUTPUT_TOKENS_EXCEEDED
        ),
        types.StopReason.MAX_OUTPUT_TOKENS_EXCEEDED,
    )
    self.assertEqual(
        event_processor._parse_stop_reason(  # pylint: disable=protected-access
            localharness_pb2.TrajectoryStateUpdate.STOP_REASON_MAX_TOTAL_TOKENS_EXCEEDED
        ),
        types.StopReason.MAX_TOTAL_TOKENS_EXCEEDED,
    )
    self.assertEqual(
        event_processor._parse_stop_reason(  # pylint: disable=protected-access
            localharness_pb2.TrajectoryStateUpdate.STOP_REASON_QUOTA_EXHAUSTED
        ),
        types.StopReason.QUOTA_EXHAUSTED,
    )
    self.assertEqual(
        event_processor._parse_stop_reason(  # pylint: disable=protected-access
            localharness_pb2.TrajectoryStateUpdate.STOP_REASON_UNSPECIFIED
        ),
        types.StopReason.UNSPECIFIED,
    )


class LocalConnectionStepFromDictTest(absltest.TestCase):
  """Tests for LocalConnectionStep.from_dict derivation logic.

  Specifically targets the is_complete_response calculation and edge cases in
  step type detection.
  """

  def test_is_complete_response_true(self):
    """Verifies is_complete_response is True when source=MODEL, state=DONE, target=TARGET_USER, and text is present.

    Why: This is the canonical "agent finished speaking" signal that callers
    rely on to surface the final answer. All four conditions must hold:
    source is MODEL, status is DONE, text is present, and target is USER.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "text": "Here is my answer.",
        "target": "TARGET_USER",
    })
    self.assertTrue(step.is_complete_response)

  def test_is_complete_response_false_when_source_not_model(self):
    """Verifies is_complete_response is False when source is not MODEL.

    Why: System or user steps that are done and have text should not be
    treated as a completed model response.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_USER",
        "state": "STATE_DONE",
        "text": "Some user text.",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_not_done(self):
    """Verifies is_complete_response is False when state is not DONE.

    Why: An active model step is still streaming; it should not be treated
    as complete until the harness marks it done.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ACTIVE",
        "text": "Partial response...",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_no_text(self):
    """Verifies is_complete_response is False when text is empty.

    Why: A done model step with no text is a structural step (e.g. tool use
    completion), not a completed textual response.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_error_state(self):
    """Verifies is_complete_response is False when state is ERROR."""
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ERROR",
        "text": "Something went wrong",
        "error_message": "internal error",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_target_environment(self):
    """Verifies is_complete_response is False for TARGET_ENVIRONMENT steps.

    Why: Tool execution steps (view_file, run_command, etc.) are targeted at
    the environment, not the user. Even when they are source=MODEL, state=DONE,
    and have text (e.g. "Requesting permission to make tool call"), they must
    not be treated as a completed model response.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "text": "Requesting permission to make tool call",
        "target": "TARGET_ENVIRONMENT",
    })
    self.assertFalse(step.is_complete_response)

  def test_step_type_tool_call_with_builtin(self):
    """Verifies that a step with a builtin tool proto field is typed TOOL_CALL and parses details."""
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ACTIVE",
        "view_file": {"file_path": "/foo"},
    })
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

    self.assertLen(step.tool_calls, 1)
    self.assertEqual(step.tool_calls[0].name, "view_file")
    self.assertEqual(step.tool_calls[0].args, {"file_path": "/foo"})
    self.assertEqual(step.tool_calls[0].step_id, "0")
    self.assertEqual(step.tool_calls[0].canonical_path, "/foo")

  def test_step_type_tool_call_with_generate_image_normalizes_output_path(self):
    """Verifies that a step with generate_image built-in tool normalizes output_path."""
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "generate_image": {
            "prompt": "A sunset",
            "image_name": "sunset",
            "aspect_ratio": "16:9",
            "output_path": "file:///tmp/sunset_123.png",
        },
    })
    self.assertEqual(step.type, types.StepType.TOOL_CALL)
    self.assertLen(step.tool_calls, 1)
    self.assertEqual(step.tool_calls[0].name, "generate_image")
    self.assertEqual(
        step.tool_calls[0].args,
        {
            "prompt": "A sunset",
            "image_name": "sunset",
            "aspect_ratio": "16:9",
            "output_path": "/tmp/sunset_123.png",
        },
    )
    self.assertEqual(step.tool_calls[0].canonical_path, "/tmp/sunset_123.png")

  def test_generate_image_result_model_output_path(self):
    """Verifies GenerateImageResult field and string output formatting."""
    res = local_types.GenerateImageResult(
        image_name="sunset",
        aspect_ratio="16:9",
        output_path="/tmp/sunset_123.png",
    )
    self.assertEqual(res.image_name, "sunset")
    self.assertEqual(res.aspect_ratio, "16:9")
    self.assertEqual(res.output_path, "/tmp/sunset_123.png")
    self.assertEqual(str(res), "/tmp/sunset_123.png")

  def test_generate_image_result_model_fallback_str(self):
    res = local_types.GenerateImageResult(
        image_name="sunset", aspect_ratio="16:9"
    )
    self.assertEqual(str(res), "sunset")

  def test_structured_output_extracted_from_finish(self):
    """Verifies that structured output is extracted when finish payload is present.

    Why: The connection layer is responsible for extracting and parsing
    the final structured output from the wire format so Layer 2 and E2E tests
    can access it natively.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "finish": {
            "output_string": (
                '{"total_revenue": 386.0, "top_selling_product": "Widget A"}'
            ),
        },
    })
    self.assertEqual(
        step.structured_output,
        {"total_revenue": 386.0, "top_selling_product": "Widget A"},
    )

  def test_structured_output_extracted_from_finish_handles_invalid_json(self):
    """Verifies that invalid JSON in finish payload defaults to None.

    Why: The connection layer should handle malformed JSON payloads gracefully
    by returning None instead of raising a fatal exception.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "finish": {
            "output_string": (  # Invalid JSON
                '{"total_revenue": 386.0, "top_selling_product": }'
            ),
        },
    })
    self.assertIsNone(step.structured_output)

  def test_step_from_dict_normalizes_file_uri_arguments(self):
    """Verifies that LocalConnectionStep.from_dict normalizes file:// URIs."""
    step = event_processor.LocalConnectionStep.from_dict({
        "step_index": 1,
        "trajectory_id": "traj_1",
        "state": "STATE_WAITING_FOR_USER",
        "view_file": {"file_path": "file:///dev/shm/workspace/foo.py"},
    })
    self.assertLen(step.tool_calls, 1)
    self.assertEqual(
        step.tool_calls[0].args.get("file_path"), "/dev/shm/workspace/foo.py"
    )
    self.assertNotIn("canonical_path", step.tool_calls[0].args)
    self.assertEqual(
        step.tool_calls[0].canonical_path,
        "/dev/shm/workspace/foo.py",
    )

  def test_step_from_dict_normalizes_cns_uri_arguments(self):
    """Verifies that LocalConnectionStep.from_dict normalizes cns:// URIs.

    Why: The CNS-backed filesystem uses cns:// URIs as path representations.
    The workspace_only policy compares canonical_path against /cns/... paths
    provided by the user, so cns:// must be translated to /cns/... for
    policy matching to work correctly.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "step_index": 1,
        "trajectory_id": "traj_1",
        "state": "STATE_WAITING_FOR_USER",
        "create_file": {"path": "cns://el-d/home/user/workspace/kittens.md"},
    })
    self.assertLen(step.tool_calls, 1)
    self.assertEqual(
        step.tool_calls[0].args.get("path"),
        "/cns/el-d/home/user/workspace/kittens.md",
    )
    self.assertNotIn("canonical_path", step.tool_calls[0].args)
    self.assertEqual(
        step.tool_calls[0].canonical_path,
        "/cns/el-d/home/user/workspace/kittens.md",
    )

  def test_step_from_dict_parses_parent_trajectory_id_and_depth(self):
    """Verifies that from_dict parses parent_trajectory_id and depth."""
    step = event_processor.LocalConnectionStep.from_dict({
        "step_index": 5,
        "trajectory_id": "child_traj_123",
        "parent_trajectory_id": "parent_traj_456",
        "depth": 2,
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
        "text": "Task finished",
    })
    self.assertEqual(step.step_index, 5)
    self.assertEqual(step.trajectory_id, "child_traj_123")
    self.assertEqual(step.parent_trajectory_id, "parent_traj_456")
    self.assertEqual(step.depth, 2)
    self.assertEqual(step.content, "Task finished")

  def test_step_type_tool_call_with_custom_tool(self):
    """Verifies that a step with a custom_tool field is typed TOOL_CALL and parses details."""
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "custom_tool": {
            "tool_call": {
                "id": "call_1",
                "name": "my_custom_tool",
                "arguments_json": (
                    '{"arg1": "val1", "file_path": "file:///foo"}'
                ),
            },
            "tool_response": {
                "id": "my_custom_tool",
                "response_json": '{"result": "ok"}',
            },
        },
    })
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

    self.assertLen(step.tool_calls, 1)
    self.assertEqual(step.tool_calls[0].name, "my_custom_tool")
    self.assertEqual(
        step.tool_calls[0].args,
        {"arg1": "val1", "file_path": "/foo"},
    )
    self.assertEqual(step.tool_calls[0].canonical_path, "/foo")
    self.assertEqual(step.tool_calls[0].id, "call_1")

  def test_step_type_tool_call_with_custom_tool_fallback_id(self):
    """Verifies that fallback ID is used if custom_tool.tool_call.id is missing."""
    step = event_processor.LocalConnectionStep.from_dict({
        "trajectory_id": "traj_123",
        "step_index": 5,
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "custom_tool": {
            "tool_call": {
                "name": "my_custom_tool",
                "arguments_json": "{}",
            },
        },
    })
    self.assertLen(step.tool_calls, 1)
    self.assertEqual(step.tool_calls[0].id, "traj_123:5")


class LocalHarnessEventProcessorTest(unittest.IsolatedAsyncioTestCase):
  """Tests for LocalHarnessEventProcessor."""

  async def test_process_event_updates_cumulative_usage(self):
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = "main_traj"

    event = localharness_pb2.OutputEvent(
        usage_update=localharness_pb2.UsageUpdate(
            agents=[
                localharness_pb2.TrajectoryUsageEntry(
                    trajectory_id="main_traj",
                    usage=localharness_pb2.UsageMetadata(
                        prompt_token_count=120, total_token_count=150
                    ),
                )
            ],
            total=localharness_pb2.UsageMetadata(
                prompt_token_count=120, total_token_count=150
            ),
        )
    )
    await processor.process_event(event)

    self.assertEqual(processor.cumulative_usage.prompt_token_count, 120)
    self.assertEqual(processor.cumulative_usage.total_token_count, 150)
    self.assertEqual(
        processor.trajectory_usages["main_traj"].prompt_token_count, 120
    )
    self.assertEqual(
        processor.trajectory_usages["main_traj"].total_token_count, 150
    )

    # Verify usage update with subagent also updates cumulative usage.
    subagent_event = localharness_pb2.OutputEvent(
        usage_update=localharness_pb2.UsageUpdate(
            agents=[
                localharness_pb2.TrajectoryUsageEntry(
                    trajectory_id="main_traj",
                    usage=localharness_pb2.UsageMetadata(
                        prompt_token_count=120, total_token_count=150
                    ),
                ),
                localharness_pb2.TrajectoryUsageEntry(
                    trajectory_id="subagent_1",
                    usage=localharness_pb2.UsageMetadata(
                        prompt_token_count=180, total_token_count=250
                    ),
                ),
            ],
            total=localharness_pb2.UsageMetadata(
                prompt_token_count=300, total_token_count=400
            ),
        )
    )
    await processor.process_event(subagent_event)

    self.assertEqual(processor.cumulative_usage.prompt_token_count, 300)
    self.assertEqual(processor.cumulative_usage.total_token_count, 400)
    self.assertEqual(
        processor.trajectory_usages["main_traj"].prompt_token_count, 120
    )
    self.assertEqual(
        processor.trajectory_usages["subagent_1"].prompt_token_count, 180
    )
    self.assertEqual(
        processor.trajectory_usages["subagent_1"].total_token_count, 250
    )

  async def test_main_agent_running_clears_idle_state(self):
    """Verifies that when the main agent is RUNNING, the connection is not idle."""
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID
    processor.is_idle.set()

    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_RUNNING,
            trajectory_id=MAIN_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)

    self.assertFalse(processor.is_idle.is_set())

  async def test_process_event_updates_stop_reason(self):
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID

    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_FULLY_IDLE,
            trajectory_id=MAIN_TRAJECTORY_ID,
            stop_reason=localharness_pb2.TrajectoryStateUpdate.STOP_REASON_QUOTA_EXHAUSTED,
        )
    )
    await processor.process_event(event)

    self.assertEqual(
        processor._last_turn_stop_reason,
        types.StopReason.QUOTA_EXHAUSTED,
    )

    processor.reset_for_turn()
    self.assertEqual(
        processor._last_turn_stop_reason,
        types.StopReason.UNSPECIFIED,
    )

  async def test_main_agent_idle_sets_idle_state(self):
    """Verifies that when the main agent is IDLE, the connection is idle."""
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID
    processor.is_idle.clear()

    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_FULLY_IDLE,
            trajectory_id=MAIN_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)

    self.assertTrue(processor.is_idle.is_set())
    self.assertEqual(processor.step_queue.qsize(), 1)
    self.assertIs(
        await processor.step_queue.get(), event_processor.IDLE_SENTINEL
    )

  async def test_main_agent_idle_with_error_sets_idle_state(self):
    """Verifies that when the main agent goes IDLE with an error, the error is enqueued before the sentinel."""
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID
    processor.is_idle.clear()

    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_FULLY_IDLE,
            trajectory_id=MAIN_TRAJECTORY_ID,
            error="Failed turn execution",
        )
    )
    await processor.process_event(event)

    self.assertTrue(processor.is_idle.is_set())
    self.assertEqual(processor.step_queue.qsize(), 2)  # error + IDLE_SENTINEL
    err = await processor.step_queue.get()
    self.assertIsInstance(err, types.AntigravityExecutionError)
    self.assertEqual(str(err), "Failed turn execution")
    self.assertIs(
        await processor.step_queue.get(), event_processor.IDLE_SENTINEL
    )

  async def test_main_agent_cancelled_sets_idle_state(self):
    """Verifies that when the main agent is CANCELLED, the connection is idle with error."""
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID
    processor.is_idle.clear()

    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_CANCELLED,
            trajectory_id=MAIN_TRAJECTORY_ID,
            error="Cancelled by user",
        )
    )
    await processor.process_event(event)

    self.assertTrue(processor.is_idle.is_set())
    self.assertEqual(processor.step_queue.qsize(), 2)  # error + IDLE_SENTINEL
    err = await processor.step_queue.get()
    self.assertIsInstance(err, types.AntigravityExecutionError)
    self.assertEqual(str(err), "Cancelled by user")
    self.assertIs(
        await processor.step_queue.get(), event_processor.IDLE_SENTINEL
    )

  async def test_subagent_state_ignored_for_idle(self):
    """Verifies that subagent state updates do not directly affect connection idleness."""
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID
    processor.is_idle.clear()

    # Subagent running shouldn't affect anything
    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_RUNNING,
            trajectory_id=SUBAGENT_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)
    self.assertFalse(processor.is_idle.is_set())

    # Subagent idle shouldn't affect anything
    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_FULLY_IDLE,
            trajectory_id=SUBAGENT_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)
    self.assertFalse(processor.is_idle.is_set())
    self.assertTrue(processor.step_queue.empty())

  @mock.patch.object(event_processor, "logging")
  async def test_subagent_error_logged_but_ignored_for_idle(self, mock_logging):
    """Verifies that subagent failures log errors but do not affect idle state."""
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID
    processor.is_idle.clear()

    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_FULLY_IDLE,
            trajectory_id=SUBAGENT_TRAJECTORY_ID,
            error="Subagent failure",
        )
    )
    await processor.process_event(event)

    mock_logging.info.assert_called_once_with(
        "Subagent trajectory failed with error: %s", "Subagent failure"
    )
    self.assertFalse(processor.is_idle.is_set())
    self.assertTrue(processor.step_queue.empty())

  async def test_process_event_skips_local_custom_tool_in_step_update(self):
    """Verifies that process_event removes custom_tool if it is a local tool."""
    mock_tool_runner = mock.MagicMock()
    mock_tool_runner.tool_names = ["my_local_tool"]
    mock_hook_runner = mock.AsyncMock()
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock(),
        tool_runner=mock_tool_runner,
        hook_runner=mock_hook_runner,
    )

    step_update_pb = localharness_pb2.StepUpdate()
    json_format.ParseDict(
        {
            "trajectory_id": "traj_123",
            "step_index": 5,
            "source": "SOURCE_MODEL",
            "state": "STATE_DONE",
            "custom_tool": {
                "tool_call": {
                    "name": "my_local_tool",
                    "arguments_json": "{}",
                },
            },
        },
        step_update_pb,
    )
    event = localharness_pb2.OutputEvent(step_update=step_update_pb)

    await processor.process_event(event)

    self.assertEqual(processor.step_queue.qsize(), 1)
    step = await processor.step_queue.get()
    self.assertEqual(len(step.tool_calls), 0)
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

    mock_hook_runner.dispatch_pre_step.assert_called_once()
    called_step = mock_hook_runner.dispatch_pre_step.call_args[0][1]
    self.assertEqual(called_step.type, types.StepType.TOOL_CALL)
    self.assertEqual(len(called_step.tool_calls), 1)
    self.assertEqual(called_step.tool_calls[0].name, "my_local_tool")

  async def test_process_event_does_not_skip_remote_custom_tool_in_step_update(
      self,
  ):
    """Verifies that process_event keeps custom_tool if it is not a local tool."""
    mock_tool_runner = mock.MagicMock()
    mock_tool_runner.tool_names = ["some_other_tool"]
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock(),
        tool_runner=mock_tool_runner,
    )

    step_update_pb = localharness_pb2.StepUpdate()
    json_format.ParseDict(
        {
            "trajectory_id": "traj_123",
            "step_index": 5,
            "source": "SOURCE_MODEL",
            "state": "STATE_DONE",
            "custom_tool": {
                "tool_call": {
                    "name": "my_remote_tool",
                    "arguments_json": "{}",
                },
            },
        },
        step_update_pb,
    )
    event = localharness_pb2.OutputEvent(step_update=step_update_pb)

    await processor.process_event(event)

    self.assertEqual(processor.step_queue.qsize(), 1)
    step = await processor.step_queue.get()
    self.assertEqual(len(step.tool_calls), 1)
    self.assertEqual(step.tool_calls[0].name, "my_remote_tool")
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

  async def test_send_tool_results_with_proto_extensions_dict(self):
    """Verifies _send_tool_results handles ToolResult containing dict with proto extensions."""
    send_mock = mock.AsyncMock()
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=send_mock
    )
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/123",
    )
    tool_result = types.ToolResult(
        id="call_123",
        name="test_tool",
        result={"output": "done", "video": video},
    )
    await processor._send_tool_results([tool_result])

    send_mock.assert_called_once()
    input_event = send_mock.call_args[0][0]
    resp = input_event.tool_response
    self.assertEqual(resp.id, "call_123")
    self.assertTrue(resp.HasField("response"))
    fields_map = {f.name: f.value for f in resp.response.fields}
    self.assertEqual(fields_map["output"].string_value, "done")
    self.assertEqual(
        fields_map["video"].content_value.video.uri,
        "https://example.com/video/123",
    )
    self.assertEqual(
        json.loads(resp.response_json)["video"],
        {"$ref": "https://example.com/video/123"},
    )

  async def test_send_tool_results_with_proto_extensions_non_dict(self):
    """Verifies _send_tool_results handles ToolResult containing non-dict with proto extensions."""
    send_mock = mock.AsyncMock()
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=send_mock
    )
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/123",
    )
    tool_result = types.ToolResult(
        id="call_456",
        name="test_tool",
        result=video,
    )
    await processor._send_tool_results([tool_result])

    send_mock.assert_called_once()
    input_event = send_mock.call_args[0][0]
    resp = input_event.tool_response
    self.assertEqual(resp.id, "call_456")
    self.assertTrue(resp.HasField("response"))
    fields_map = {f.name: f.value for f in resp.response.fields}
    self.assertEqual(
        fields_map["result"].content_value.video.uri,
        "https://example.com/video/123",
    )
    self.assertEqual(
        json.loads(resp.response_json)["result"],
        {"$ref": "https://example.com/video/123"},
    )

  async def test_send_tool_results_with_proto_extensions_pydantic_model(self):
    """Verifies _send_tool_results handles ToolResult containing Pydantic model with proto extensions without wrapping."""

    class ToolOutput(pydantic.BaseModel):
      model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
      output: str
      video: Any

    send_mock = mock.AsyncMock()
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=send_mock
    )
    video = content_pb2.VideoContent(
        mime_type=content_pb2.VideoContent.TYPE_MP4,
        uri="https://example.com/video/123",
    )
    tool_result = types.ToolResult(
        id="call_789",
        name="test_tool",
        result=ToolOutput(output="done", video=video),
    )
    await processor._send_tool_results([tool_result])

    send_mock.assert_called_once()
    input_event = send_mock.call_args[0][0]
    resp = input_event.tool_response
    self.assertEqual(resp.id, "call_789")
    self.assertTrue(resp.HasField("response"))
    fields_map = {f.name: f.value for f in resp.response.fields}
    self.assertEqual(fields_map["output"].string_value, "done")
    self.assertEqual(
        fields_map["video"].content_value.video.uri,
        "https://example.com/video/123",
    )
    self.assertEqual(
        json.loads(resp.response_json)["video"],
        {"$ref": "https://example.com/video/123"},
    )


def _make_policy_decision_request(
    request_id: str = "req-1",
    rule_id: str = "rule_0",
    tool_name: str = "run_command",
    arguments_json: str = "{}",
    server_name: str = "",
) -> localharness_pb2.OutputEvent:
  return localharness_pb2.OutputEvent(
      policy_decision_request=localharness_pb2.PolicyDecisionRequest(
          request_id=request_id,
          rule_id=rule_id,
          tool_args=localharness_pb2.PreToolArgs(
              tool_name=tool_name,
              arguments_json=arguments_json,
              server_name=server_name,
          ),
      )
  )


class PolicyDecisionTest(unittest.IsolatedAsyncioTestCase):
  """Tests for PolicyDecisionRequest handling in the event processor."""

  async def _process_and_get_response(
      self, dynamic_policy_map, event
  ) -> localharness_pb2.PolicyDecisionResponse:
    """Helper: creates a processor, processes an event, returns the response."""
    send_mock = mock.AsyncMock()
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=send_mock,
        dynamic_policy_map=dynamic_policy_map,
    )
    await processor.process_event(event)
    # Let the background task complete.
    await asyncio.sleep(0.01)
    send_mock.assert_called_once()
    input_event = send_mock.call_args[0][0]
    return input_event.policy_decision_response

  async def test_when_predicate_matches_deny(self):
    """when returns True + DENY -> OUTCOME_DENY."""
    p = policy.deny("run_command", when=lambda args: True, name="block-all")
    _, pmap = policy._to_policy_config_proto([p])
    event = _make_policy_decision_request(rule_id="rule_0")
    resp = await self._process_and_get_response(pmap, event)
    self.assertEqual(resp.request_id, "req-1")
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_DENY
    )
    self.assertIn("block-all", resp.deny_reason)

  async def test_when_predicate_no_match(self):
    """when returns False -> OUTCOME_NO_MATCH."""
    p = policy.deny("run_command", when=lambda args: False)
    _, pmap = policy._to_policy_config_proto([p])
    event = _make_policy_decision_request(rule_id="rule_0")
    resp = await self._process_and_get_response(pmap, event)
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_NO_MATCH
    )

  async def test_when_predicate_matches_allow(self):
    """when returns True + ALLOW -> OUTCOME_ALLOW."""
    p = policy.allow("run_command", when=lambda args: True)
    _, pmap = policy._to_policy_config_proto([p])
    event = _make_policy_decision_request(rule_id="rule_0")
    resp = await self._process_and_get_response(pmap, event)
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_ALLOW
    )

  async def test_ask_user_approve(self):
    """ASK_USER handler returns True -> OUTCOME_ALLOW."""
    p = policy.ask_user("run_command", handler=lambda tc: True, name="ask-test")
    _, pmap = policy._to_policy_config_proto([p])
    event = _make_policy_decision_request(rule_id="rule_0")
    resp = await self._process_and_get_response(pmap, event)
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_ALLOW
    )
    self.assertEqual(resp.deny_reason, "")

  async def test_ask_user_deny(self):
    """ASK_USER handler returns False -> OUTCOME_DENY."""
    p = policy.ask_user(
        "run_command", handler=lambda tc: False, name="ask-deny"
    )
    _, pmap = policy._to_policy_config_proto([p])
    event = _make_policy_decision_request(rule_id="rule_0")
    resp = await self._process_and_get_response(pmap, event)
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_DENY
    )
    self.assertIn("Denied by user", resp.deny_reason)

  async def test_ask_user_with_when_no_match(self):
    """ASK_USER + when=False -> OUTCOME_NO_MATCH (handler never called)."""
    handler_mock = mock.Mock(return_value=True)
    p = policy.ask_user(
        "run_command", handler=handler_mock, when=lambda args: False
    )
    _, pmap = policy._to_policy_config_proto([p])
    event = _make_policy_decision_request(rule_id="rule_0")
    resp = await self._process_and_get_response(pmap, event)
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_NO_MATCH
    )
    handler_mock.assert_not_called()

  async def test_unknown_rule_id_fails_closed(self):
    """Unknown rule_id -> OUTCOME_DENY."""
    event = _make_policy_decision_request(rule_id="nonexistent")
    resp = await self._process_and_get_response({}, event)
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_DENY
    )
    self.assertIn("Unknown rule_id", resp.deny_reason)

  async def test_predicate_exception_fails_closed(self):
    """Predicate raises exception -> OUTCOME_DENY."""

    def bad_predicate(args):
      raise ValueError("boom")

    p = policy.deny("run_command", when=bad_predicate)
    _, pmap = policy._to_policy_config_proto([p])
    event = _make_policy_decision_request(rule_id="rule_0")
    resp = await self._process_and_get_response(pmap, event)
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_DENY
    )
    self.assertIn("Policy evaluation error", resp.deny_reason)

  async def test_async_ask_user_handler(self):
    """Async ask_user handler works."""

    async def async_handler(tc):
      return True

    p = policy.ask_user("run_command", handler=async_handler)
    _, pmap = policy._to_policy_config_proto([p])
    event = _make_policy_decision_request(rule_id="rule_0")
    resp = await self._process_and_get_response(pmap, event)
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_ALLOW
    )

  async def test_when_predicate_receives_args(self):
    """Predicate receives parsed tool arguments."""
    captured = {}

    def capture_predicate(args):
      captured.update(args)
      return True

    p = policy.deny("run_command", when=capture_predicate)
    _, pmap = policy._to_policy_config_proto([p])
    event = _make_policy_decision_request(
        rule_id="rule_0",
        arguments_json=json.dumps({"CommandLine": "echo hello"}),
    )
    resp = await self._process_and_get_response(pmap, event)
    self.assertEqual(
        resp.outcome, localharness_pb2.POLICY_EVALUATION_OUTCOME_DENY
    )
    self.assertEqual(captured.get("CommandLine"), "echo hello")


if __name__ == "__main__":
  absltest.main()
