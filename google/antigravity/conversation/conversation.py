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

"""Stateful conversation session for the Google Antigravity SDK.

Conversation is the Layer 2 session API. It wraps a Connection with:
- Step history accumulation (with compaction index tracking)
- A chat() convenience method (send + collect in one call)
- State introspection (idle, turn count, last response)

Layer 1 (Agent) delegates to Conversation; power users can use
Conversation directly with any ConnectionStrategy.
"""

import contextlib
from typing import Any, AsyncIterator, Sequence

from google.antigravity import types
from google.antigravity.connections import connection as connection_lib

# Default maximum number of steps to retain in history.
_DEFAULT_MAX_HISTORY_SIZE = 10_000


class Conversation:
  """Stateful session wrapping a single conversation with the agent.

  Accumulates step history, tracks turn start indices and compaction indices,
  and provides convenience methods for common send/receive patterns.
  """

  def __init__(
      self,
      conn: connection_lib.Connection,
      *,
      max_history_size: int = _DEFAULT_MAX_HISTORY_SIZE,
      history: Sequence[types.Step] | None = None,
  ):
    """Initializes the conversation with a connection and optional initial history.

    Args:
      conn: The established connection to the agent backend.
      max_history_size: Maximum number of steps to retain in history.
        When exceeded, the oldest steps are discarded. Set to 0 to
        disable the limit.
      history: Optional initial sequence of steps to seed the conversation
        history.
    """
    self._connection = conn
    self._steps: list[types.Step] = []
    self._turn_start_indices: list[int] = []
    self._compaction_indices: list[int] = []
    self._max_history_size = max_history_size
    self._turn_start_usage: types.UsageMetadata | None = None
    for step in history or []:
      self._steps.append(step)
      if step.type == types.StepType.COMPACTION:
        self._compaction_indices.append(len(self._steps) - 1)
    self._enforce_max_history()

  @classmethod
  @contextlib.asynccontextmanager
  async def create(
      cls,
      strategy: connection_lib.ConnectionStrategy,
  ) -> AsyncIterator["Conversation"]:
    """Creates a new conversation.

    Args:
      strategy: The connection strategy to use to interact with an agent.

    Yields:
      A new Conversation instance.
    """
    async with strategy:
      conn = strategy.connect()
      yield cls(conn, history=conn._initial_history)

  # ---------------------------------------------------------------------------
  # Core send / receive
  # ---------------------------------------------------------------------------

  async def send(
      self,
      prompt: types.Content | None,
      **kwargs: Any,
  ) -> None:
    """Sends a message to the agent.

    If a turn is still in progress, drains all remaining steps into
    history before sending the new message.  When another coroutine is
    already iterating ``receive_steps()`` (and therefore recording steps
    into history), falls back to ``wait_for_idle()``.

    Args:
      prompt: The user message to send.
      **kwargs: Strategy-specific options.
    """
    if self._connection.is_idle is False:
      try:
        async for _ in self.receive_steps():
          pass
      except RuntimeError:
        # Catches the async generator "already running" error from
        # Python's runtime.  Intentionally broad: any RuntimeError here
        # means we cannot drain, so falling back to wait_for_idle() is
        # safe since the active iterator is already preserving steps.
        await self._connection.wait_for_idle()
    self._turn_start_indices.append(len(self._steps))
    self._turn_start_usage = self._connection.cumulative_usage.model_copy()
    await self._connection.send(prompt, **kwargs)

  async def receive_steps(self) -> AsyncIterator[types.Step]:
    """Receives steps as they complete, blocks until execution is idle.

    Steps are recorded in history as they arrive. The iterator exits once
    the execution turn is complete.

    Returns:
      An async iterator of Step objects.

    Yields:
      Steps as they complete.
    """
    async for step in self._connection.receive_steps():
      self._steps.append(step)
      if step.type == types.StepType.COMPACTION:
        self._compaction_indices.append(len(self._steps) - 1)
      self._enforce_max_history()
      yield step

  async def receive_chunks(
      self,
  ) -> AsyncIterator[types.StreamChunk | types.ToolCall]:
    """Receives and yields real-time semantic chunks for the current turn.

    Returns:
      An async iterator of Thought, Text, or ToolCall events.

    Yields:
      Thought, Text, or ToolCall events in real-time.
    """
    seen_tool_ids: set[str] = set()
    async for step in self.receive_steps():
      is_model = step.source == types.StepSource.MODEL
      is_target_user = step.target == types.StepTarget.USER

      if is_model and is_target_user:
        # Yield real-time thought deltas directly
        if step.thinking_delta:
          yield types.Thought(
              step_index=step.step_index, text=step.thinking_delta
          )

        # Yield real-time text deltas directly
        if step.content_delta:
          yield types.Text(step_index=step.step_index, text=step.content_delta)

      # Yield tool calls in real-time, deduplicating across steps.
      # The agentic loop can emit the same ToolCall in multiple step
      # transitions (dispatch, execution, result).  Calls without an
      # ID are always yielded since we cannot determine duplicates.
      if step.tool_calls:
        for call in step.tool_calls:
          if call.id is None or call.id not in seen_tool_ids:
            if call.id is not None:
              seen_tool_ids.add(call.id)
            yield call

  def get_last_structured_output(self) -> Any | None:
    """Extracts the structured output payload from the most recent FINISH step.

    Returns:
      The parsed JSON structured output object, or None if not found.
    """
    for step in reversed(self._steps):
      if step.type == types.StepType.FINISH:
        return step.structured_output
    return None

  async def chat(
      self, prompt: types.Content | None = None, **kwargs: Any
  ) -> types.ChatResponse:
    """Sends a prompt and returns a streaming ChatResponse instantly.

    This is a unified entry point supporting real-time text delta streaming
    via iteration, or non-blocking deferred accessors for standard reading.

    Args:
      prompt: The user message to send.
      **kwargs: Strategy-specific options.

    Returns:
      A lazy streaming ChatResponse object wrapping the chunk generator.
    """
    await self.send(prompt, **kwargs)
    return types.ChatResponse(self.receive_chunks(), conversation=self)

  # ---------------------------------------------------------------------------
  # History and state
  # ---------------------------------------------------------------------------

  @property
  def history(self) -> list[types.Step]:
    """Returns all steps received across all turns.

    This is the full, uncompacted transcript. Use compaction_indices to
    identify where the model's context window was compacted.
    """
    return list(self._steps)

  @property
  def last_response(self) -> str:
    """Returns the content of the most recent final model response."""
    for step in reversed(self._steps):
      if step.is_complete_response:
        return step.content
    return ""

  @property
  def turn_count(self) -> int:
    """Returns the number of send() calls made on this conversation."""
    return len(self._turn_start_indices)

  @property
  def compaction_indices(self) -> list[int]:
    """Step indices where the model's context was compacted.

    Each index corresponds to the position of a compaction step in
    the history list. Steps before these indices may no longer be in
    the model's context window, but remain in the full history for
    transcript and debugging.
    """
    return list(self._compaction_indices)

  def clear_history(self) -> None:
    """Clears the accumulated step history, turn indices, and compaction indices.

    Use this to free memory during long-running sessions. The conversation
    remains active — only the recorded history is discarded.
    """
    self._steps.clear()
    self._turn_start_indices.clear()
    self._compaction_indices.clear()
    self._turn_start_usage = None

  def _enforce_max_history(self) -> None:
    """Trims history to max_history_size if a limit is set."""
    if self._max_history_size and len(self._steps) > self._max_history_size:
      overflow = len(self._steps) - self._max_history_size
      self._steps = self._steps[overflow:]
      # Adjust indices to account for removed steps.
      self._turn_start_indices = [
          i - overflow
          for i in self._turn_start_indices
          if i >= overflow
      ]
      self._compaction_indices = [
          i - overflow
          for i in self._compaction_indices
          if i >= overflow
      ]

  @property
  def connection(self) -> connection_lib.Connection:
    """Returns the underlying Connection transport.

    Intended for advanced use cases that need direct transport access.
    Prefer Conversation or Agent methods for normal interaction —
    bypassing the Conversation layer skips history tracking and hook
    dispatch.
    """
    return self._connection

  @property
  def is_idle(self) -> bool:
    """Returns True if the conversation is idle and ready for input."""
    return self._connection.is_idle

  @property
  def conversation_id(self) -> str:
    """Returns the conversation identifier, if one exists."""
    return self._connection.conversation_id

  @property
  def total_usage(self) -> types.UsageMetadata:
    """Returns cumulative token usage across all turns in this session."""
    return self._connection.cumulative_usage.model_copy()

  @property
  def trajectory_usages(self) -> dict[str, types.UsageMetadata]:
    """Returns a dictionary mapping trajectory ID to cumulative token usage.

    The main agent's trajectory ID matches conversation_id (or 'main').
    Subagents each have a unique trajectory ID.
    """
    return {
        k: v.model_copy() for k, v in self._connection.trajectory_usages.items()
    }

  @property
  def last_turn_usage(self) -> types.UsageMetadata | None:
    """Returns token usage from the most recent turn, or None."""
    if self._turn_start_usage is None:
      return None
    diff = self._connection.cumulative_usage - self._turn_start_usage
    if diff.total_token_count == 0:
      return None
    return diff

  @property
  def _last_turn_usage(self) -> types.UsageMetadata | None:
    """Internal alias for last_turn_usage."""
    return self.last_turn_usage

  @property
  def _last_turn_stop_reason(self) -> types.StopReason:
    """Returns the stop reason of the most recent turn."""
    return self._connection._last_turn_stop_reason  # pylint: disable=protected-access

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  async def cancel(self) -> None:
    """Cancels the current turn in progress."""
    await self._connection.cancel()

  async def wait_for_idle(self) -> None:
    """Blocks until the conversation is idle and ready for the next turn."""
    await self._connection.wait_for_idle()

  async def wait_for_wakeup(self, timeout: float = 300.0) -> bool:
    """Blocks until the conversation wakes up or the timeout is reached.

    Args:
      timeout: Maximum seconds to wait.

    Returns:
      True if the conversation woke up, False on timeout.
    """
    return await self._connection.wait_for_wakeup(timeout)

  async def disconnect(self) -> None:
    """Closes the connection transport and releases background resources."""
    await self._connection.disconnect()
