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

"""Tests for the Conversation stateful session layer.

Validates history accumulation, compaction tracking, chat() convenience,
state introspection, and clean delegation to the Connection ABC.
"""

import unittest
from unittest import mock

from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.conversation import conversation


def _make_step(
    content: str = "",
    *,
    step_index: int = 0,
    step_type: types.StepType = types.StepType.TEXT_RESPONSE,
    source: types.StepSource = types.StepSource.MODEL,
    status: types.StepStatus = types.StepStatus.DONE,
    is_final: bool = False,
    target: types.StepTarget = types.StepTarget.USER,
) -> types.Step:
  """Creates a Step with sensible defaults for testing."""
  step = types.Step(
      id=str(step_index),
      step_index=step_index,
      type=step_type,
      source=source,
      status=status,
      content=content,
      is_complete_response=is_final,
  )
  step.target = target
  return step


class ConversationCreateTest(unittest.IsolatedAsyncioTestCase):
  """Validates the Conversation.create factory."""

  async def test_create_delegates_to_strategy(self):
    """Verifies that create enters the strategy context and calls connect."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_strategy = mock.AsyncMock(spec=connection.ConnectionStrategy)
    mock_strategy.connect.return_value = mock_connection

    async with conversation.Conversation.create(mock_strategy) as conv:
      self.assertIsInstance(conv, conversation.Conversation)

    mock_strategy.connect.assert_called_once()


class ConversationSendTest(unittest.IsolatedAsyncioTestCase):
  """Validates send behavior including idle-wait and turn tracking."""

  async def test_send_when_idle_delegates_directly(self):
    """When already idle, send() delegates to connection without draining."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    type(mock_connection).is_idle = mock.PropertyMock(return_value=True)
    conv = conversation.Conversation(mock_connection)

    await conv.send("hello")

    mock_connection.send.assert_called_once_with("hello")
    mock_connection.receive_steps.assert_not_called()

  async def test_send_multimodal_input(self):
    """Verifies that send accepts multimodal Content payloads and delegates to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    type(mock_connection).is_idle = mock.PropertyMock(return_value=True)
    conv = conversation.Conversation(mock_connection)

    multimodal_prompt = [
        "Context string",
        types.Document(mime_type="application/pdf", data=b"pdf"),
    ]
    await conv.send(multimodal_prompt)

    mock_connection.send.assert_called_once_with(multimodal_prompt)

  async def test_send_records_turn_boundary(self):
    """Verifies each send records a turn boundary index in the history."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    type(mock_connection).is_idle = mock.PropertyMock(return_value=True)
    conv = conversation.Conversation(mock_connection)

    await conv.send("first")
    self.assertEqual(conv.turn_count, 1)

    await conv.send("second")
    self.assertEqual(conv.turn_count, 2)


class ConversationReceiveStepsTest(unittest.IsolatedAsyncioTestCase):
  """Validates receive_steps delegation and history accumulation."""

  async def test_receive_steps_yields_from_connection(self):
    """Verifies steps are yielded from the underlying connection."""
    step1 = _make_step("step1", step_index=1)
    step2 = _make_step("step2", step_index=2)

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield step1
      yield step2

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    steps = []
    async for step in conv.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 2)
    self.assertEqual(steps[0].content, "step1")
    self.assertEqual(steps[1].content, "step2")

  async def test_receive_steps_accumulates_history(self):
    """Verifies steps are recorded in history as they are received."""
    step1 = _make_step("a", step_index=1)
    step2 = _make_step("b", step_index=2)

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield step1
      yield step2

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    # History starts empty.
    self.assertEqual(conv.history, [])

    async for _ in conv.receive_steps():
      pass

    self.assertEqual(len(conv.history), 2)
    self.assertEqual(conv.history[0].content, "a")
    self.assertEqual(conv.history[1].content, "b")

  async def test_history_returns_copy(self):
    """Verifies history returns a copy, not a reference to internal state."""
    step = _make_step("x", step_index=1)
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    async for _ in conv.receive_steps():
      pass

    history = conv.history
    history.clear()
    self.assertEqual(len(conv.history), 1)

  async def test_compaction_step_tracked(self):
    """Verifies compaction steps are recorded in compaction_indices."""
    regular = _make_step("text", step_index=1)
    compaction = _make_step(
        "compacted",
        step_index=2,
        step_type=types.StepType.COMPACTION,
    )
    after = _make_step("more", step_index=3)

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield regular
      yield compaction
      yield after

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    async for _ in conv.receive_steps():
      pass

    # Compaction step is at index 1 in the history list.
    self.assertEqual(conv.compaction_indices, [1])
    # Full history is preserved.
    self.assertEqual(len(conv.history), 3)

  async def test_compaction_indices_returns_copy(self):
    """Verifies compaction_indices returns a copy."""
    compaction = _make_step(
        "",
        step_index=1,
        step_type=types.StepType.COMPACTION,
    )
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield compaction

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    async for _ in conv.receive_steps():
      pass

    indices = conv.compaction_indices
    indices.clear()
    self.assertEqual(len(conv.compaction_indices), 1)

  async def test_compaction_streaming_active_then_done_tracks_only_done(self):
    """Verifies only completed (DONE) compaction steps enter compaction_indices."""
    regular = _make_step("text", step_index=1)
    compacting = _make_step(
        "Context compaction",
        step_index=2,
        step_type=types.StepType.COMPACTION,
        source=types.StepSource.SYSTEM,
        status=types.StepStatus.ACTIVE,
    )
    compacted = _make_step(
        "Context compaction",
        step_index=2,
        step_type=types.StepType.COMPACTION,
        source=types.StepSource.SYSTEM,
        status=types.StepStatus.DONE,
    )
    after = _make_step("more", step_index=3)

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield regular
      yield compacting
      yield compacted
      yield after

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    async for _ in conv.receive_steps():
      pass

    self.assertEqual(conv.compaction_indices, [2])
    self.assertEqual(len(conv.history), 4)

    seeded = conversation.Conversation(
        mock_connection, history=[regular, compacting, compacted, after]
    )
    self.assertEqual(seeded.compaction_indices, [2])

  async def test_compaction_aborted_active_not_tracked(self):
    """Verifies an un-stopped ACTIVE compaction step is excluded from compaction_indices."""
    compacting = _make_step(
        "Context compaction",
        step_index=1,
        step_type=types.StepType.COMPACTION,
        source=types.StepSource.SYSTEM,
        status=types.StepStatus.ACTIVE,
    )
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield compacting

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    async for _ in conv.receive_steps():
      pass

    self.assertEqual(conv.compaction_indices, [])
    self.assertEqual(len(conv.history), 1)

  async def test_compaction_subagent_excluded_from_compaction_indices(self):
    """Verifies completed subagent COMPACTION steps do not enter root compaction_indices."""
    root_step = _make_step("root work", step_index=1)
    child_compacted = _make_step(
        "Context compaction",
        step_index=1,
        step_type=types.StepType.COMPACTION,
        source=types.StepSource.SYSTEM,
        status=types.StepStatus.DONE,
    )
    child_compacted.depth = 1
    child_compacted.trajectory_id = "sub-1"
    child_compacted.parent_trajectory_id = "root"
    root_compacted = _make_step(
        "Context compaction",
        step_index=2,
        step_type=types.StepType.COMPACTION,
        source=types.StepSource.SYSTEM,
        status=types.StepStatus.DONE,
    )

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield root_step
      yield child_compacted
      yield root_compacted

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    async for _ in conv.receive_steps():
      pass

    self.assertEqual(conv.compaction_indices, [2])
    self.assertEqual(len(conv.history), 3)

    seeded = conversation.Conversation(
        mock_connection, history=[root_step, child_compacted, root_compacted]
    )
    self.assertEqual(seeded.compaction_indices, [2])


class ConversationReceiveChunksTest(unittest.IsolatedAsyncioTestCase):
  """Tests covering receive_chunks text, thought, and tool-call delta routing."""

  async def test_receive_chunks_routes_thoughts(self):
    """Verifies that receive_chunks correctly yields Thought deltas."""
    step = _make_step(
        "Thinking phase", step_index=1, status=types.StepStatus.ACTIVE
    )
    step.thinking_delta = "Thinking..."

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    chunks = [chunk async for chunk in conv.receive_chunks()]

    self.assertEqual(len(chunks), 1)
    self.assertIsInstance(chunks[0], types.Thought)
    self.assertEqual(chunks[0].text, "Thinking...")

  async def test_receive_chunks_routes_text(self):
    """Verifies that receive_chunks correctly yields Text deltas."""
    s1 = _make_step(
        "Answer start", step_index=1, status=types.StepStatus.ACTIVE
    )
    s1.content_delta = "Hello"

    s2 = _make_step("Answer end", step_index=1, status=types.StepStatus.DONE)
    s2.content_delta = " world!"

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield s1
      yield s2

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    chunks = [chunk async for chunk in conv.receive_chunks()]

    self.assertEqual(len(chunks), 2)
    self.assertIsInstance(chunks[0], types.Text)
    self.assertEqual(chunks[0].text, "Hello")
    self.assertIsInstance(chunks[1], types.Text)
    self.assertEqual(chunks[1].text, " world!")

  async def test_receive_chunks_filters_out_telemetry_noise(self) -> None:
    """Verifies that receive_chunks ignores prompts and environmental tool checks."""
    s_prompt = _make_step(
        "Prompt context", step_index=0, source=types.StepSource.USER
    )
    s_confirm = _make_step(
        "Confirming tool call...",
        step_index=1,
        target=types.StepTarget.ENVIRONMENT,
    )
    s_valid = _make_step(
        "valid output",
        step_index=2,
        source=types.StepSource.MODEL,
        target=types.StepTarget.USER,
    )
    s_valid.content_delta = "Valid answer"

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield s_prompt
      yield s_confirm
      yield s_valid

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    chunks = [chunk async for chunk in conv.receive_chunks()]

    self.assertEqual(len(chunks), 1)
    self.assertIsInstance(chunks[0], types.Text)
    self.assertEqual(chunks[0].text, "Valid answer")

  async def test_receive_chunks_filters_out_error_steps(self) -> None:
    """Verifies that receive_chunks does not yield text or thoughts from ERROR steps."""
    s_error = _make_step(
        "Cannot view file file:///tmp/missing.txt which does not exist.",
        step_index=1,
        source=types.StepSource.MODEL,
        target=types.StepTarget.USER,
        status=types.StepStatus.ERROR,
    )
    s_error.content_delta = (
        "Cannot view file file:///tmp/missing.txt which does not exist."
    )
    s_error.thinking_delta = "Error thinking"

    s_valid = _make_step(
        "Final answer",
        step_index=2,
        source=types.StepSource.MODEL,
        target=types.StepTarget.USER,
        status=types.StepStatus.DONE,
    )
    s_valid.content_delta = "Final answer"

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield s_error
      yield s_valid

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    chunks = [chunk async for chunk in conv.receive_chunks()]

    self.assertEqual(len(chunks), 1)
    self.assertIsInstance(chunks[0], types.Text)
    self.assertEqual(chunks[0].text, "Final answer")

  async def test_receive_chunks_routes_tool_calls(self) -> None:
    """Verifies that receive_chunks yields strongly-typed ToolCall objects natively."""
    tc = types.ToolCall(
        id="call_123", name="view_file", args={"path": "README.md"}
    )
    s_tool = _make_step("", step_index=1, step_type=types.StepType.TOOL_CALL)
    s_tool.tool_calls = [tc]

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield s_tool

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    chunks = [chunk async for chunk in conv.receive_chunks()]

    self.assertEqual(len(chunks), 1)
    self.assertIsInstance(chunks[0], types.ToolCall)
    self.assertEqual(chunks[0].name, "view_file")
    self.assertEqual(chunks[0].args, {"path": "README.md"})

  async def test_receive_chunks_deduplicates_tool_calls(self) -> None:
    """Verifies that the same ToolCall emitted across multiple steps is yielded only once."""
    tc = types.ToolCall(
        id="call_456", name="generate_image", args={"prompt": "cat"}
    )
    # The agentic loop can emit the same tool call in dispatch,
    # execution, and result steps.
    s_dispatch = _make_step(
        "", step_index=1, step_type=types.StepType.TOOL_CALL
    )
    s_dispatch.tool_calls = [tc]

    s_exec = _make_step(
        "", step_index=2, step_type=types.StepType.TOOL_CALL
    )
    s_exec.tool_calls = [tc]

    s_result = _make_step(
        "", step_index=3, step_type=types.StepType.TOOL_CALL
    )
    s_result.tool_calls = [tc]

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield s_dispatch
      yield s_exec
      yield s_result

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    chunks = [chunk async for chunk in conv.receive_chunks()]

    tool_calls = [c for c in chunks if isinstance(c, types.ToolCall)]
    self.assertEqual(len(tool_calls), 1)
    self.assertEqual(tool_calls[0].id, "call_456")

  async def test_receive_chunks_yields_distinct_tool_calls(self) -> None:
    """Verifies that different tool calls with different IDs are all yielded."""
    tc1 = types.ToolCall(id="call_a", name="tool_1", args={})
    tc2 = types.ToolCall(id="call_b", name="tool_2", args={})

    step = _make_step("", step_index=1, step_type=types.StepType.TOOL_CALL)
    step.tool_calls = [tc1, tc2]

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    chunks = [chunk async for chunk in conv.receive_chunks()]

    tool_calls = [c for c in chunks if isinstance(c, types.ToolCall)]
    self.assertEqual(len(tool_calls), 2)
    self.assertEqual(tool_calls[0].id, "call_a")
    self.assertEqual(tool_calls[1].id, "call_b")

  async def test_receive_chunks_never_deduplicates_none_id_calls(self) -> None:
    """Verifies that tool calls with id=None are always yielded."""
    tc1 = types.ToolCall(id=None, name="tool_x", args={"a": 1})
    tc2 = types.ToolCall(id=None, name="tool_x", args={"a": 2})

    s1 = _make_step("", step_index=1, step_type=types.StepType.TOOL_CALL)
    s1.tool_calls = [tc1]
    s2 = _make_step("", step_index=2, step_type=types.StepType.TOOL_CALL)
    s2.tool_calls = [tc2]

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield s1
      yield s2

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    chunks = [chunk async for chunk in conv.receive_chunks()]

    tool_calls = [c for c in chunks if isinstance(c, types.ToolCall)]
    self.assertEqual(len(tool_calls), 2)

  async def test_receive_chunks_never_deduplicates_empty_string_id_calls(
      self,
  ) -> None:
    """Verifies that tool calls with id='' (empty string) are always yielded."""
    tc1 = types.ToolCall(id="", name="tool_x", args={"a": 1})
    tc2 = types.ToolCall(id="", name="tool_x", args={"a": 2})

    s1 = _make_step("", step_index=1, step_type=types.StepType.TOOL_CALL)
    s1.tool_calls = [tc1]
    s2 = _make_step("", step_index=2, step_type=types.StepType.TOOL_CALL)
    s2.tool_calls = [tc2]

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield s1
      yield s2

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    chunks = [chunk async for chunk in conv.receive_chunks()]

    tool_calls = [c for c in chunks if isinstance(c, types.ToolCall)]
    self.assertEqual(len(tool_calls), 2)


class ConversationHistoryTest(unittest.IsolatedAsyncioTestCase):
  """Validates history accessors across multiple turns."""

  async def test_last_response_returns_most_recent_final(self):
    """Verifies last_response returns the content of the most recent final step."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def gen1():
      yield _make_step("first answer", step_index=1, is_final=True)

    async def gen2():
      yield _make_step("second answer", step_index=2, is_final=True)

    mock_connection.receive_steps.return_value = gen1()
    conv = conversation.Conversation(mock_connection)

    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()
    await conv.send("q1")
    async for _ in conv.receive_steps():
      pass

    self.assertEqual(conv.last_response, "first answer")

    mock_connection.receive_steps.return_value = gen2()
    await conv.send("q2")
    async for _ in conv.receive_steps():
      pass

    self.assertEqual(conv.last_response, "second answer")

  async def test_last_response_empty_when_no_final(self):
    """Verifies last_response returns empty string when no final step exists."""
    conv = conversation.Conversation(mock.MagicMock(spec=connection.Connection))
    self.assertEqual(conv.last_response, "")

  async def test_multi_turn_history_accumulates(self):
    """Verifies history accumulates across multiple send/receive cycles."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def gen1():
      yield _make_step("a", step_index=1)

    async def gen2():
      yield _make_step("b", step_index=2)
      yield _make_step("c", step_index=3)

    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()
    conv = conversation.Conversation(mock_connection)

    mock_connection.receive_steps.return_value = gen1()
    await conv.send("turn1")
    async for _ in conv.receive_steps():
      pass

    mock_connection.receive_steps.return_value = gen2()
    await conv.send("turn2")
    async for _ in conv.receive_steps():
      pass

    self.assertEqual(len(conv.history), 3)
    self.assertEqual(conv.turn_count, 2)


class ConversationChatTest(unittest.IsolatedAsyncioTestCase):
  """Validates the chat() convenience method."""

  async def test_chat_returns_streaming_response_with_text(self):
    """Verifies chat() returns ChatResponse, and text() resolves the final string."""
    tool_step = _make_step("", step_index=1, step_type=types.StepType.TOOL_CALL)
    final_step = _make_step("the answer", step_index=2, is_final=True)
    final_step.content_delta = "the answer"

    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield tool_step
      yield final_step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    result = await conv.chat("question")

    self.assertIsInstance(result, types.ChatResponse)
    self.assertEqual(await result.text(), "the answer")

  async def test_chat_multimodal_input(self):
    """Verifies that the chat convenience wrapper accepts and forwards multimodal Content prompts."""
    final_step = _make_step("image analysis done", step_index=1, is_final=True)
    final_step.content_delta = "image analysis done"
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield final_step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    multimodal_prompt = [
        "Analyze this blueprint:",
        types.Image(mime_type="image/png", data=b"blueprint_bytes"),
    ]
    result = await conv.chat(multimodal_prompt)

    self.assertEqual(await result.text(), "image analysis done")
    mock_connection.send.assert_called_once_with(multimodal_prompt)

  async def test_chat_records_in_history(self):
    """Verifies chat() steps are accumulated in conversation history after resolution."""
    step = _make_step("done", step_index=1, is_final=True)
    step.content_delta = "done"
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    result = await conv.chat("q")
    await result.resolve()

    self.assertEqual(len(conv.history), 1)
    self.assertEqual(conv.turn_count, 1)

  async def test_chat_empty_response_when_no_final(self):
    """Verifies chat() returns empty text when no final response step exists."""
    step = _make_step("interim", step_index=1, is_final=False)
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    result = await conv.chat("q")

    self.assertEqual(await result.text(), "")

  async def test_chat_returns_structured_output_when_final_step_has_it(self):
    """Verifies chat() collects and returns structured_output via lazy accessor."""
    final_step = _make_step(
        "done", step_index=1, step_type=types.StepType.FINISH, is_final=True
    )
    final_step.structured_output = {"total_revenue": 386.0}

    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield final_step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    result = await conv.chat("question")

    self.assertEqual(await result.structured_output(), {"total_revenue": 386.0})


class ConversationMultiTurnIsolationTest(unittest.IsolatedAsyncioTestCase):
  """Validates per-turn ChatResponse state isolation across multi-turn sessions."""

  def _make_scripted_connection(
      self,
      turns: list[
          tuple[
              list[types.Step],
              types.UsageMetadata,
              types.StopReason,
          ]
      ],
  ) -> mock.MagicMock:
    """Builds a Connection mock whose turns advance deterministically on send().

    Args:
      turns: Sequence of (steps, turn_usage_delta, stop_reason) tuples, one per
        turn in order.

    Returns:
      A configured Connection mock.
    """
    mock_conn = mock.MagicMock(spec=connection.Connection)
    mock_conn.wait_for_idle = mock.AsyncMock()
    mock_conn.cumulative_usage = types.UsageMetadata()
    mock_conn._last_turn_stop_reason = types.StopReason.UNSPECIFIED
    state = {"turn_idx": -1, "idle": True}

    type(mock_conn).is_idle = mock.PropertyMock(
        side_effect=lambda: state["idle"]
    )

    async def send_side_effect(prompt, **kwargs):
      del prompt, kwargs
      state["turn_idx"] += 1
      state["idle"] = False

    def receive_steps_side_effect():
      idx = state["turn_idx"]
      steps, usage_delta, stop_reason = turns[idx]

      async def _gen():
        for step in steps:
          yield step
        mock_conn.cumulative_usage = mock_conn.cumulative_usage + usage_delta
        mock_conn._last_turn_stop_reason = stop_reason
        state["idle"] = True

      return _gen()

    mock_conn.send = mock.AsyncMock(side_effect=send_side_effect)
    mock_conn.receive_steps.side_effect = receive_steps_side_effect
    return mock_conn

  async def test_multi_turn_chat_isolates_structured_output_usage_and_stop_reason(
      self,
  ):
    """Verifies Turn 2 does not clobber Turn 1's ChatResponse state.

    What: Runs two structured-output turns with distinct payloads, token usages,
      and stop reasons on a single Conversation.
    Why: Ensures retaining earlier ChatResponse references across multi-turn
      conversations preserves each turn's own structured_output(),
      usage_metadata, and stop_reason.
    How: Resolves Turn 1, runs and resolves Turn 2, and asserts Turn 1's
      accessors still return Turn 1's exact values alongside Turn 2's values.
    """
    s1_text = _make_step("Turn 1 text", step_index=1, is_final=True)
    s1_text.content_delta = "Turn 1 text"
    s1_finish = _make_step(
        "Finished", step_index=2, step_type=types.StepType.FINISH
    )
    s1_finish.structured_output = {
        "service_name": "payments-gateway",
        "estimated_rps": 12500,
    }

    s2_text = _make_step("Turn 2 text", step_index=3, is_final=True)
    s2_text.content_delta = "Turn 2 text"
    s2_finish = _make_step(
        "Finished", step_index=4, step_type=types.StepType.FINISH
    )
    s2_finish.structured_output = {
        "service_name": "internal-ledger",
        "estimated_rps": 400,
    }

    usage1 = types.UsageMetadata(
        prompt_token_count=100,
        cached_content_token_count=0,
        candidates_token_count=20,
        thoughts_token_count=10,
        total_token_count=130,
    )
    usage2 = types.UsageMetadata(
        prompt_token_count=250,
        cached_content_token_count=0,
        candidates_token_count=35,
        thoughts_token_count=15,
        total_token_count=300,
    )
    mock_conn = self._make_scripted_connection([
        ([s1_text, s1_finish], usage1, types.StopReason.UNSPECIFIED),
        (
            [s2_text, s2_finish],
            usage2,
            types.StopReason.MAX_MODEL_CALLS_EXCEEDED,
        ),
    ])
    conv = conversation.Conversation(mock_conn)

    resp1 = await conv.chat("Turn 1")
    self.assertEqual(
        await resp1.structured_output(),
        {"service_name": "payments-gateway", "estimated_rps": 12500},
    )
    self.assertEqual(resp1.usage_metadata, usage1)
    self.assertEqual(resp1.stop_reason, types.StopReason.UNSPECIFIED)

    resp2 = await conv.chat("Turn 2")
    self.assertEqual(
        await resp2.structured_output(),
        {"service_name": "internal-ledger", "estimated_rps": 400},
    )
    self.assertEqual(resp2.usage_metadata, usage2)
    self.assertEqual(
        resp2.stop_reason, types.StopReason.MAX_MODEL_CALLS_EXCEEDED
    )

    # Re-inspect Turn 1 after Turn 2 completed.
    self.assertEqual(await resp1.text(), "Turn 1 text")
    self.assertEqual(
        await resp1.structured_output(),
        {"service_name": "payments-gateway", "estimated_rps": 12500},
    )
    self.assertEqual(resp1.usage_metadata, usage1)
    self.assertEqual(resp1.stop_reason, types.StopReason.UNSPECIFIED)
    self.assertEqual(conv.total_usage.total_token_count, 430)

  async def test_mixed_structured_and_unstructured_turns_do_not_bleed(self):
    """Verifies unstructured or subagent-only turns do not inherit prior output.

    What: Runs Turn 1 with structured_output, Turn 2 with plain text plus
      subagent FINISH steps, Turn 3 with structured_output=None, and Turn 4 with
      falsy structured_output={}.
    Why: Prevents a turn without a root FINISH step from inheriting an earlier
      turn's FINISH step or picking up a child subagent's FINISH step, while
      preserving falsy JSON payloads like {}.
    How: Asserts each turn's ChatResponse.structured_output() matches strictly
      its own root FINISH payload (or None when absent).
    """
    t1_finish = _make_step(
        "Finished", step_index=1, step_type=types.StepType.FINISH
    )
    t1_finish.structured_output = {"turn": 1}

    t2_subagent_depth_only = _make_step(
        "Subagent depth", step_index=1, step_type=types.StepType.FINISH
    )
    t2_subagent_depth_only.depth = 1
    t2_subagent_depth_only.structured_output = {"subagent_depth": 98}

    t2_subagent_parent_only = _make_step(
        "Subagent parent", step_index=2, step_type=types.StepType.FINISH
    )
    t2_subagent_parent_only.parent_trajectory_id = "root"
    t2_subagent_parent_only.structured_output = {"subagent_parent": 99}

    t2_plain = _make_step("Plain reply", step_index=3, is_final=True)
    t2_plain.content_delta = "Plain reply"

    t3_empty_finish = _make_step(
        "", step_index=4, step_type=types.StepType.FINISH
    )
    t3_empty_finish.structured_output = None

    t4_falsy_finish = _make_step(
        "{}", step_index=5, step_type=types.StepType.FINISH
    )
    t4_falsy_finish.structured_output = {}

    zero_usage = types.UsageMetadata(
        prompt_token_count=0,
        cached_content_token_count=0,
        candidates_token_count=0,
        thoughts_token_count=0,
        total_token_count=0,
    )
    mock_conn = self._make_scripted_connection([
        ([t1_finish], zero_usage, types.StopReason.UNSPECIFIED),
        (
            [t2_subagent_depth_only, t2_subagent_parent_only, t2_plain],
            zero_usage,
            types.StopReason.UNSPECIFIED,
        ),
        ([t3_empty_finish], zero_usage, types.StopReason.UNSPECIFIED),
        ([t4_falsy_finish], zero_usage, types.StopReason.UNSPECIFIED),
    ])
    conv = conversation.Conversation(mock_conn)

    resp1 = await conv.chat("Turn 1")
    self.assertEqual(await resp1.structured_output(), {"turn": 1})

    resp2 = await conv.chat("Turn 2")
    self.assertEqual(await resp2.text(), "Plain reply")
    self.assertIsNone(await resp2.structured_output())
    self.assertEqual(await resp1.structured_output(), {"turn": 1})

    resp3 = await conv.chat("Turn 3")
    self.assertIsNone(await resp3.structured_output())
    self.assertEqual(await resp1.structured_output(), {"turn": 1})

    resp4 = await conv.chat("Turn 4")
    self.assertEqual(await resp4.structured_output(), {})

    seeded = conversation.Conversation(
        mock_conn,
        history=[t1_finish, t2_subagent_depth_only, t2_subagent_parent_only],
    )
    self.assertEqual(seeded.get_last_structured_output(), {"turn": 1})

  async def test_max_history_trimming_and_clear_history_preserve_turn_snapshots(
      self,
  ):
    """Verifies max_history_size trimming and clear_history() keep turn state.

    What: Tests FINISH step evicted from history mid-turn by max_history_size=2
      and clear_history() between turns.
    Why: Ensures memory-bounding history operations never drop a turn's
      structured_output, usage_metadata, or stop_reason.
    How: Asserts structured_output() and metadata survive history eviction and
      clear_history().
    """
    finish1 = _make_step(
        "Finished", step_index=1, step_type=types.StepType.FINISH
    )
    finish1.structured_output = {"trimmed_turn": 1}
    trailing_steps = [
        _make_step(f"post-{i}", step_index=i + 2, is_final=(i == 2))
        for i in range(3)
    ]
    trailing_steps[-1].content_delta = "done"

    usage1 = types.UsageMetadata(
        prompt_token_count=25,
        cached_content_token_count=0,
        candidates_token_count=0,
        thoughts_token_count=0,
        total_token_count=25,
    )
    mock_conn = self._make_scripted_connection([
        (
            [finish1, *trailing_steps],
            usage1,
            types.StopReason.MAX_OUTPUT_TOKENS_EXCEEDED,
        ),
    ])
    conv = conversation.Conversation(mock_conn, max_history_size=2)

    resp1 = await conv.chat("Turn 1")
    self.assertEqual(await resp1.text(), "done")
    self.assertEqual(len(conv.history), 2)
    self.assertEqual(await resp1.structured_output(), {"trimmed_turn": 1})

    conv.clear_history()
    self.assertIsNone(conv.get_last_structured_output())
    self.assertEqual(await resp1.structured_output(), {"trimmed_turn": 1})
    self.assertEqual(resp1.usage_metadata, usage1)
    self.assertEqual(
        resp1.stop_reason, types.StopReason.MAX_OUTPUT_TOKENS_EXCEEDED
    )


class ConversationStateTest(unittest.IsolatedAsyncioTestCase):
  """Validates state introspection properties."""

  async def test_is_idle_delegates_to_connection(self):
    """Verifies is_idle reads from connection.is_idle property."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    type(mock_connection).is_idle = mock.PropertyMock(return_value=False)
    conv = conversation.Conversation(mock_connection)

    self.assertFalse(conv.is_idle)

    type(mock_connection).is_idle = mock.PropertyMock(return_value=True)
    self.assertTrue(conv.is_idle)

  async def test_conversation_id_delegates_to_connection(self):
    """Verifies conversation_id reads from connection.conversation_id."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    type(mock_connection).conversation_id = mock.PropertyMock(
        return_value="conv-123"
    )
    conv = conversation.Conversation(mock_connection)

    self.assertEqual(conv.conversation_id, "conv-123")

  async def test_conversation_id_empty_by_default(self):
    """Verifies conversation_id returns empty string from default ABC impl."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    type(mock_connection).conversation_id = mock.PropertyMock(return_value="")
    conv = conversation.Conversation(mock_connection)

    self.assertEqual(conv.conversation_id, "")

  async def test_connection_returns_underlying_transport(self):
    """Verifies connection property returns the injected Connection."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)

    self.assertIs(conv.connection, mock_connection)


class ConversationLifecycleTest(unittest.IsolatedAsyncioTestCase):
  """Validates direct delegation of lifecycle methods without hasattr guards."""

  async def test_cancel_delegates(self):
    """Verifies cancel delegates directly to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)
    await conv.cancel()
    mock_connection.cancel.assert_called_once()

  async def test_wait_for_idle_delegates(self):
    """Verifies wait_for_idle delegates directly to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)
    await conv.wait_for_idle()
    mock_connection.wait_for_idle.assert_called_once()

  async def test_wait_for_wakeup_delegates(self):
    """Verifies wait_for_wakeup delegates to connection with timeout."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    mock_connection.wait_for_wakeup.return_value = True
    conv = conversation.Conversation(mock_connection)
    result = await conv.wait_for_wakeup(timeout=60.0)
    self.assertTrue(result)
    mock_connection.wait_for_wakeup.assert_called_once_with(60.0)

  async def test_disconnect_delegates(self):
    """Verifies disconnect delegates directly to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)
    await conv.disconnect()
    mock_connection.disconnect.assert_called_once()


class ConversationClearHistoryTest(unittest.IsolatedAsyncioTestCase):
  """Validates clear_history and max_history_size behavior."""

  async def test_clear_history_resets_all_state(self):
    """Verifies clear_history empties steps, turns, and compaction indices."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    compaction = _make_step(
        "", step_index=1, step_type=types.StepType.COMPACTION,
    )
    final = _make_step("answer", step_index=2, is_final=True)

    async def mock_generator():
      yield compaction
      yield final

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    await conv.send("q")
    async for _ in conv.receive_steps():
      pass

    # Verify state is populated.
    self.assertEqual(len(conv.history), 2)
    self.assertEqual(conv.turn_count, 1)
    self.assertEqual(len(conv.compaction_indices), 1)

    conv.clear_history()

    self.assertEqual(conv.history, [])
    self.assertEqual(conv.turn_count, 0)
    self.assertEqual(conv.compaction_indices, [])
    self.assertEqual(conv.last_response, "")

  async def test_max_history_trims_oldest_steps(self):
    """Verifies max_history_size trims oldest steps when exceeded."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      for i in range(10):
        yield _make_step(f"step-{i}", step_index=i)

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection, max_history_size=5)

    async for _ in conv.receive_steps():
      pass

    self.assertEqual(len(conv.history), 5)
    # Oldest steps are trimmed; newest remain.
    self.assertEqual(conv.history[0].content, "step-5")
    self.assertEqual(conv.history[-1].content, "step-9")

  async def test_max_history_adjusts_compaction_indices(self):
    """Verifies compaction indices are adjusted when history is trimmed."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield _make_step("a", step_index=0)
      yield _make_step(
          "", step_index=1, step_type=types.StepType.COMPACTION,
      )
      yield _make_step("b", step_index=2)
      yield _make_step("c", step_index=3)
      yield _make_step("d", step_index=4)

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection, max_history_size=3)

    async for _ in conv.receive_steps():
      pass

    # History trimmed to last 3 steps: [b, c, d].
    self.assertEqual(len(conv.history), 3)
    # Compaction at original index 1 was before the trim window, so removed.
    self.assertEqual(conv.compaction_indices, [])

  async def test_max_history_zero_disables_limit(self):
    """Verifies max_history_size=0 disables the limit."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      for i in range(100):
        yield _make_step(f"step-{i}", step_index=i)

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection, max_history_size=0)

    async for _ in conv.receive_steps():
      pass

    self.assertEqual(len(conv.history), 100)


class ConversationUsageMetadataTest(unittest.IsolatedAsyncioTestCase):

  def _make_step_with_usage(
      self,
      step_index: int = 0,
      prompt: int | None = None,
      candidates: int | None = None,
      total: int | None = None,
      thoughts: int | None = None,
      cached: int | None = None,
      is_final: bool = False,
  ) -> types.Step:
    """Creates a Step with optional usage_metadata for testing."""
    usage = types.UsageMetadata(
        prompt_token_count=prompt,
        cached_content_token_count=cached,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        total_token_count=total,
    )
    return types.Step(
        id=str(step_index),
        step_index=step_index,
        type=types.StepType.TEXT_RESPONSE,
        source=types.StepSource.MODEL,
        status=types.StepStatus.DONE,
        content="",
        is_complete_response=is_final,
        usage_metadata=usage,
    )

  def _make_conv_with_mock(self):
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()
    # Provide a mutable UsageMetadata that tests can update to simulate
    # the backend pushing cumulative_usage via TrajectoryStateUpdate.
    mock_connection.cumulative_usage = types.UsageMetadata(
        prompt_token_count=0,
        cached_content_token_count=0,
        candidates_token_count=0,
        thoughts_token_count=0,
        total_token_count=0,
    )
    return conversation.Conversation(mock_connection), mock_connection

  async def test_total_usage_starts_at_zero(self):
    """Verifies cumulative usage is initialized to zero, not None."""
    conv, _ = self._make_conv_with_mock()
    usage = conv.total_usage
    self.assertEqual(
        usage,
        types.UsageMetadata(
            prompt_token_count=0,
            cached_content_token_count=0,
            candidates_token_count=0,
            thoughts_token_count=0,
            total_token_count=0,
        ),
    )

  async def test_total_usage_accumulates_across_steps(self):
    """Verifies total_usage comes from cumulative_usage, per-turn via diff."""
    conv, mock_connection = self._make_conv_with_mock()

    async def gen():
      yield self._make_step_with_usage(0, prompt=100, candidates=50, total=150)
      yield self._make_step_with_usage(1, prompt=200, candidates=30, total=230)

    mock_connection.receive_steps.return_value = gen()
    await conv.send("q")
    async for _ in conv.receive_steps():
      pass

    # Simulate backend pushing session cumulative usage after all steps.
    mock_connection.cumulative_usage = types.UsageMetadata(
        prompt_token_count=300,
        cached_content_token_count=0,
        candidates_token_count=80,
        thoughts_token_count=0,
        total_token_count=380,
    )
    usage = conv.total_usage
    self.assertEqual(usage.prompt_token_count, 300)
    self.assertEqual(usage.candidates_token_count, 80)
    self.assertEqual(usage.total_token_count, 380)

    # Per-turn usage is a diff of cumulative_usage.
    self.assertEqual(conv._last_turn_usage.prompt_token_count, 300)
    self.assertEqual(conv._last_turn_usage.candidates_token_count, 80)

  async def test_total_usage_ignores_none_fields(self):
    """Verifies per-turn usage is computed via cumulative_usage diff."""
    conv, mock_connection = self._make_conv_with_mock()

    step_with = self._make_step_with_usage(0, prompt=100, thoughts=10)
    step_without = _make_step("no usage", step_index=1)  # usage_metadata=None

    async def gen():
      yield step_with
      yield step_without

    mock_connection.receive_steps.return_value = gen()
    await conv.send("q")
    async for _ in conv.receive_steps():
      pass

    # Simulate backend pushing session cumulative usage.
    mock_connection.cumulative_usage = types.UsageMetadata(
        prompt_token_count=100, thoughts_token_count=10, total_token_count=110
    )
    self.assertEqual(conv._last_turn_usage.prompt_token_count, 100)
    self.assertEqual(conv._last_turn_usage.thoughts_token_count, 10)

  async def test_total_usage_accumulates_across_turns(self):
    """Verifies cumulative usage spans multiple send/receive cycles.

    With cascade ownership, the backend pushes cumulative_usage.
    This test verifies that total_usage reflects the latest pushed value.
    """
    conv, mock_connection = self._make_conv_with_mock()

    async def gen1():
      yield self._make_step_with_usage(0, prompt=100, total=120)

    async def gen2():
      yield self._make_step_with_usage(1, prompt=150, total=180)

    mock_connection.receive_steps.return_value = gen1()
    await conv.send("turn1")
    async for _ in conv.receive_steps():
      pass

    # Simulate backend pushing usage after turn 1.
    mock_connection.cumulative_usage = types.UsageMetadata(
        prompt_token_count=100, total_token_count=120
    )

    mock_connection.receive_steps.return_value = gen2()
    await conv.send("turn2")
    async for _ in conv.receive_steps():
      pass

    # Simulate backend pushing usage after turn 2 (cumulative).
    mock_connection.cumulative_usage = types.UsageMetadata(
        prompt_token_count=250, total_token_count=300
    )

    usage = conv.total_usage
    self.assertEqual(usage.prompt_token_count, 250)
    self.assertEqual(usage.total_token_count, 300)
    self.assertEqual(conv._last_turn_usage.prompt_token_count, 150)
    self.assertEqual(conv._last_turn_usage.total_token_count, 180)

  async def test_total_usage_returns_copy(self):
    """Verifies total_usage returns a copy, not a reference to internal state."""
    conv, _ = self._make_conv_with_mock()
    usage = conv.total_usage
    usage.prompt_token_count = 999
    self.assertEqual(conv.total_usage.prompt_token_count, 0)

  async def test_clear_history_resets_usage(self):
    """Verifies clear_history resets local turn usage snapshot."""
    conv, mock_connection = self._make_conv_with_mock()

    async def gen():
      yield self._make_step_with_usage(0, prompt=500, total=600)

    mock_connection.receive_steps.return_value = gen()
    await conv.send("q")
    async for _ in conv.receive_steps():
      pass

    # Simulate backend pushing usage.
    mock_connection.cumulative_usage = types.UsageMetadata(
        prompt_token_count=500, total_token_count=600
    )
    self.assertEqual(conv.total_usage.prompt_token_count, 500)

    conv.clear_history()

    # Per-turn usage is reset locally.
    self.assertIsNone(conv._last_turn_usage)

  async def test_chat_returns_accumulated_usage_metadata(self):
    """Verifies chat() returns per-turn usage via cumulative_usage diff."""
    conv, mock_connection = self._make_conv_with_mock()

    step1 = self._make_step_with_usage(0, prompt=100, candidates=50, total=150)
    step2 = self._make_step_with_usage(
        1, prompt=200, candidates=30, total=230, is_final=True
    )
    step2.content = "the answer"
    step2.is_complete_response = True

    async def gen():
      yield step1
      yield step2

    mock_connection.receive_steps.return_value = gen()
    # Simulate backend pushing cumulative usage after send() snapshots.
    async def send_side_effect(*_args, **_kwargs):
      mock_connection.cumulative_usage = types.UsageMetadata(
          prompt_token_count=300,
          candidates_token_count=80,
          total_token_count=380,
      )

    mock_connection.send.side_effect = send_side_effect
    result = await conv.chat("question")
    await result.resolve()

    self.assertIsNotNone(result.usage_metadata)
    self.assertEqual(result.usage_metadata.prompt_token_count, 300)
    self.assertEqual(result.usage_metadata.candidates_token_count, 80)
    self.assertEqual(result.usage_metadata.total_token_count, 380)

  async def test_chat_returns_none_usage_when_absent(self):
    """Verifies chat() returns zero usage when backend hasn't pushed any."""
    conv, mock_connection = self._make_conv_with_mock()

    async def gen():
      yield _make_step("answer", step_index=0, is_final=True)

    mock_connection.receive_steps.return_value = gen()
    result = await conv.chat("question")
    await result.resolve()

    # No cumulative_usage was pushed, so turn usage is None.
    self.assertIsNone(result.usage_metadata)

  async def test_initialization_with_history_usage(self):
    """Verifies that historical steps do NOT populate total_usage (backend owns it).

    With cascade ownership, total_usage comes from the connection's
    cumulative_usage, not from per-step accumulation. Historical
    steps are loaded for display but their usage is not summed locally.
    """
    hist_step1 = self._make_step_with_usage(0, prompt=100, total=120)
    hist_step2 = self._make_step_with_usage(1, prompt=200, total=240)

    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection._initial_history = [hist_step1, hist_step2]
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()
    # Backend seeds cumulative_usage during init handshake.
    mock_connection.cumulative_usage = types.UsageMetadata(
        prompt_token_count=300, total_token_count=360
    )

    conv = conversation.Conversation(
        mock_connection, history=[hist_step1, hist_step2]
    )

    # Total cumulative usage comes from the backend, not from history steps.
    self.assertEqual(conv.total_usage.prompt_token_count, 300)
    self.assertEqual(conv.total_usage.total_token_count, 360)

    # Turn usage must be completely clean (None)
    self.assertIsNone(conv._last_turn_usage)


class ConversationSendDrainTest(unittest.IsolatedAsyncioTestCase):
  """Validates send() drain-to-history when a prior turn has not been consumed."""

  def _make_conn(self):
    """Returns a mock Connection with is_idle as a controllable property."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.send = mock.AsyncMock()
    mock_connection.wait_for_idle = mock.AsyncMock()
    return mock_connection

  async def test_back_to_back_send_drains_first_turn(self):
    """A second send() auto-drains the first turn's steps into history.

    Simulates two send/receive cycles where the caller never explicitly
    iterates receive_steps() after the first send. The second send() should
    detect that the connection is not idle, drain the pending steps from
    the first turn, and then proceed normally.
    """
    mock_connection = self._make_conn()

    step_turn1 = _make_step("first reply", step_index=1, is_final=True)
    step_turn2 = _make_step("second reply", step_index=2, is_final=True)

    # Track which generator call we're on.
    receive_call_count = 0

    async def gen_turn1():
      yield step_turn1

    async def gen_turn2():
      yield step_turn2

    def receive_steps_side_effect():
      nonlocal receive_call_count
      receive_call_count += 1
      if receive_call_count <= 2:
        # Call 1: from explicit send-drain; Call 2: from explicit iteration.
        # Both use turn1's generator initially, but we switch after send.
        return gen_turn1()
      return gen_turn2()

    mock_connection.receive_steps.side_effect = receive_steps_side_effect

    # is_idle starts False (turn in progress), then True after drain,
    # then False again (second turn in progress), then True.
    type(mock_connection).is_idle = mock.PropertyMock(
        side_effect=[False, True]
    )

    conv = conversation.Conversation(mock_connection)

    # First send: connection is idle at construction, so we configure idle=False
    # to trigger the drain path.
    await conv.send("first question")

    # Caller does NOT iterate receive_steps; immediately sends again.
    # send() sees is_idle=False, drains via receive_steps(), which captures
    # step_turn1 into history.
    await conv.send("second question")

    # At this point, the drain from the second send should have captured
    # step_turn1 into history.
    self.assertEqual(len(conv.history), 1)
    self.assertEqual(conv.history[0].content, "first reply")
    self.assertEqual(conv.turn_count, 2)

    # Now explicitly drain the second turn.
    mock_connection.receive_steps.side_effect = None
    mock_connection.receive_steps.return_value = gen_turn2()
    async for _ in conv.receive_steps():
      pass

    # Both turns should now be in history.
    self.assertEqual(len(conv.history), 2)
    self.assertEqual(conv.history[0].content, "first reply")
    self.assertEqual(conv.history[1].content, "second reply")

  async def test_send_falls_back_to_wait_for_idle_on_runtime_error(self):
    """Concurrent receive_steps() triggers wait_for_idle() fallback.

    When another coroutine is already iterating receive_steps(), send()
    catches RuntimeError and falls back to wait_for_idle().

    This covers the case where two coroutines race: one is already draining
    steps (and recording them into history), while the other tries to send
    a new prompt. The RuntimeError from the async generator signals that
    it's already in use, so send() just waits for the connection to go idle.
    """
    mock_connection = self._make_conn()

    type(mock_connection).is_idle = mock.PropertyMock(return_value=False)

    # Make receive_steps() raise RuntimeError, simulating a generator
    # that is already being iterated by another coroutine.
    async def raise_runtime_error():
      raise RuntimeError("anext: asynchronous generator is already running")
      yield  # pragma: no cover  # Makes this an async generator.  # pylint: disable=unreachable

    mock_connection.receive_steps.return_value = raise_runtime_error()

    conv = conversation.Conversation(mock_connection)
    await conv.send("prompt while draining")

    # The RuntimeError was caught; wait_for_idle was called as fallback.
    mock_connection.wait_for_idle.assert_called_once()
    # send still proceeded after the fallback.
    mock_connection.send.assert_called_once_with("prompt while draining")
    self.assertEqual(conv.turn_count, 1)

  def test_last_turn_stop_reason(self) -> None:
    mock_connection = self._make_conn()
    mock_connection._last_turn_stop_reason = types.StopReason.UNSPECIFIED
    conv = conversation.Conversation(mock_connection)
    self.assertEqual(
        conv._last_turn_stop_reason,
        types.StopReason.UNSPECIFIED,
    )

    mock_connection._last_turn_stop_reason = types.StopReason.QUOTA_EXHAUSTED
    self.assertEqual(
        conv._last_turn_stop_reason,
        types.StopReason.QUOTA_EXHAUSTED,
    )


if __name__ == "__main__":
  unittest.main()
