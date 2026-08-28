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

"""Event processor for localharness events."""

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, cast

from google.protobuf import json_format
import pydantic

from google.antigravity.proto import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local import struct_converter
from google.antigravity.connections.local import types as local_types
from google.antigravity.connections.local.hook_router import HookRouter
from google.antigravity.connections.local.local_connection_config import BUILTIN_TOOL_PROTO_FIELDS
from google.antigravity.connections.local.local_connection_config import make_step_id
from google.antigravity.connections.local.local_connection_config import normalize_wire_path
from google.antigravity.connections.local.local_connection_config import WIRE_PATH_ARGUMENT_KEYS
from google.antigravity.connections.local.proto_converters import _parse_stop_reason
from google.antigravity.hooks import hook_runner as h_runner
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy as policy_lib
from google.antigravity.tools import tool_runner as t_runner

_ANY_ADAPTER = pydantic.TypeAdapter(Any)
_MCP_TOOL_PROTO_FIELD = "mcp_tool"

IDLE_SENTINEL = object()
CLOSE_SENTINEL = None

_SOURCE_MAP = {
    "SOURCE_SYSTEM": types.StepSource.SYSTEM,
    "SOURCE_USER": types.StepSource.USER,
    "SOURCE_MODEL": types.StepSource.MODEL,
}

_STATUS_MAP = {
    "STATE_ACTIVE": types.StepStatus.ACTIVE,
    "STATE_DONE": types.StepStatus.DONE,
    "STATE_WAITING_FOR_USER": types.StepStatus.WAITING_FOR_USER,
    "STATE_ERROR": types.StepStatus.ERROR,
}

_TARGET_MAP = {
    "TARGET_USER": types.StepTarget.USER,
    "TARGET_ENVIRONMENT": types.StepTarget.ENVIRONMENT,
    "TARGET_UNSPECIFIED": types.StepTarget.UNSPECIFIED,
}


# Tool-result values of these types are delivered to the model as supplemental
# media rather than JSON-serialized into the text result.
_MEDIA_TYPES = (types.Image, types.Document, types.Audio, types.Video)


def _extract_media_from_result(value: Any) -> tuple[Any, list[Any]]:
  """Splits media attachments out of a tool result value.

  Recurses through lists and dicts, pulling out any Image/Document/Audio/Video
  so they can be delivered to the model as supplemental media rather than being
  JSON-serialized (which would just embed opaque base64 in the text result).

  Args:
    value: The tool's return value.

  Returns:
    A (cleaned_value, media) tuple: ``cleaned_value`` is ``value`` with the
    media removed (None if it was entirely media), and ``media`` is the list of
    extracted media objects in encounter order.
  """
  if isinstance(value, _MEDIA_TYPES):
    return None, [value]
  if isinstance(value, (list, tuple)):
    cleaned_list = []
    list_media = []
    for item in value:
      cleaned_item, item_media = _extract_media_from_result(item)
      list_media.extend(item_media)
      if cleaned_item is not None:
        cleaned_list.append(cleaned_item)
    return type(value)(cleaned_list) or None, list_media  # pylint: disable=too-many-function-args
  if isinstance(value, dict):
    cleaned_dict = {}
    dict_media = []
    for key, item in value.items():
      cleaned_item, item_media = _extract_media_from_result(item)
      dict_media.extend(item_media)
      if cleaned_item is not None:
        cleaned_dict[key] = cleaned_item
    return (cleaned_dict or None), dict_media
  return value, []


class _StepTracker:
  """Tracks internal state transitions of a single step to deduplicate events."""

  def __init__(self):
    self.state: localharness_pb2.StepUpdate.State = (
        localharness_pb2.StepUpdate.State.STATE_UNSPECIFIED
    )
    self.pre_step_dispatched = False
    self.post_step_dispatched = False
    self.handled_requests: set[str] = set()

  def update_state(self, new_state: localharness_pb2.StepUpdate.State) -> None:
    if (
        self.state == localharness_pb2.StepUpdate.State.STATE_WAITING_FOR_USER
        and new_state
        != localharness_pb2.StepUpdate.State.STATE_WAITING_FOR_USER
    ):
      self.handled_requests.clear()
    self.state = new_state

  def mark_handled(self, request_type: str) -> bool:
    if request_type in self.handled_requests:
      return False
    self.handled_requests.add(request_type)
    return True


_TOOL_RESULT_MODELS: dict[str, Any] = {
    types.BuiltinTools.RUN_COMMAND: local_types.RunCommandResult,
    types.BuiltinTools.LIST_DIR: local_types.ListDirectoryResult,
    types.BuiltinTools.FIND_FILE: local_types.FindFileResult,
    types.BuiltinTools.SEARCH_DIR: local_types.SearchDirectoryResult,
    types.BuiltinTools.EDIT_FILE: local_types.EditFileResult,
    types.BuiltinTools.GENERATE_IMAGE: local_types.GenerateImageResult,
    types.BuiltinTools.SEARCH_WEB: local_types.SearchWebResult,
    types.BuiltinTools.READ_URL_CONTENT: local_types.ReadUrlContentResult,
}


def _extract_tool_result(
    tool_name: str, result_str: str
) -> "local_types.ToolOutput | None":
  """Extracts a structured tool result from canonical JSON string or fallback text."""
  if not result_str or tool_name not in _TOOL_RESULT_MODELS:
    return None
  try:
    res = _TOOL_RESULT_MODELS[tool_name].model_validate_json(result_str)
    return cast("local_types.ToolOutput", res)
  except (
      ValueError,
      TypeError,
      pydantic.ValidationError,
  ):
    if tool_name == types.BuiltinTools.EDIT_FILE:
      return local_types.EditFileResult(summary=result_str)
    if tool_name == types.BuiltinTools.FIND_FILE:
      return local_types.FindFileResult(output=result_str)
    return None


_make_step_id = make_step_id


def parse_usage_metadata(
    usage_metadata: localharness_pb2.UsageMetadata,
) -> types.UsageMetadata:
  """Extracts UsageMetadata from proto message."""
  return types.UsageMetadata(
      prompt_token_count=usage_metadata.prompt_token_count
      if usage_metadata.HasField("prompt_token_count")
      else None,
      cached_content_token_count=usage_metadata.cached_content_token_count
      if usage_metadata.HasField("cached_content_token_count")
      else None,
      candidates_token_count=usage_metadata.candidates_token_count
      if usage_metadata.HasField("candidates_token_count")
      else None,
      thoughts_token_count=usage_metadata.thoughts_token_count
      if usage_metadata.HasField("thoughts_token_count")
      else None,
      total_token_count=usage_metadata.total_token_count
      if usage_metadata.HasField("total_token_count")
      else None,
      service_tier=types.ServiceTier(usage_metadata.service_tier)
      if usage_metadata.service_tier
      else None,
  )


class LocalConnectionStep(types.Step):
  """Connection-specific step for LocalConnection."""

  http_code: int = 0

  @classmethod
  def from_dict(cls, step_dict: dict[str, Any]) -> "LocalConnectionStep":
    """Creates a LocalConnectionStep from a dictionary representation of StepUpdate.

    Args:
      step_dict: Dictionary containing StepUpdate fields.

    Returns:
      A new LocalConnectionStep instance.
    """
    traj_id = step_dict.get("trajectory_id", "")
    parent_traj_id = step_dict.get("parent_trajectory_id", "")
    depth_val = step_dict.get("depth", 0)
    step_idx = step_dict.get("step_index", 0)

    id_str = _make_step_id(traj_id, step_idx)

    tool_calls = []

    # Find the active built-in tool enum and field name, if any.
    active_tool_pair = next(
        (
            (tool_enum.value, step_dict[proto_field])
            for tool_enum, proto_field in BUILTIN_TOOL_PROTO_FIELDS.items()
            if proto_field in step_dict
        ),
        (None, {}),
    )
    active_tool_name, sub_msg = active_tool_pair
    active_tool_args = sub_msg if isinstance(sub_msg, dict) else {}

    active_server_name = None
    active_tool_id = None
    # Reconstruct the step's tool name and arguments from the Go-native McpTool
    # proto format to maintain Python-side trajectory parity.
    if not active_tool_name and _MCP_TOOL_PROTO_FIELD in step_dict:
      mcp_dict = step_dict[_MCP_TOOL_PROTO_FIELD]
      if isinstance(mcp_dict, dict):
        server_name = mcp_dict.get("server_name", "")
        tool_name = mcp_dict.get("tool_name", "")
        active_tool_name = tool_name
        active_server_name = server_name
        arguments_json = mcp_dict.get("arguments_json") or "{}"
        active_tool_args = json.loads(arguments_json)

    if not active_tool_name and "custom_tool" in step_dict:
      ct_dict = step_dict["custom_tool"]
      if isinstance(ct_dict, dict) and "tool_call" in ct_dict:
        tc_dict = ct_dict["tool_call"]
        if isinstance(tc_dict, dict):
          active_tool_name = tc_dict.get("name", "")
          active_tool_id = tc_dict.get("id")
          arguments_json = tc_dict.get("arguments_json") or "{}"
          try:
            active_tool_args = json.loads(arguments_json)
          except json.JSONDecodeError:
            active_tool_args = {}

    if active_tool_name:
      canonical_path = None
      # Sanitize all known file path argument fields in-place
      for path_key in WIRE_PATH_ARGUMENT_KEYS:
        if path_key in active_tool_args and isinstance(
            active_tool_args[path_key], str
        ):
          normalized = normalize_wire_path(active_tool_args[path_key])
          active_tool_args[path_key] = normalized
          canonical_path = normalized

      tool_calls.append(
          types.ToolCall(
              name=active_tool_name,
              args=active_tool_args,
              id=active_tool_id or _make_step_id(traj_id, step_idx),
              step_id=_make_step_id(traj_id, step_idx),
              canonical_path=canonical_path,
              server_name=active_server_name,
          )
      )

    # Determine high-level type
    step_type = types.StepType.UNKNOWN
    if step_dict.get("compaction") is not None:
      step_type = types.StepType.COMPACTION
    elif step_dict.get("finish") is not None:
      step_type = types.StepType.FINISH
    elif active_tool_name or any(
        step_dict.get(k) is not None for k in BUILTIN_TOOL_PROTO_FIELDS.values()
    ):
      step_type = types.StepType.TOOL_CALL
    elif step_dict.get("text"):
      step_type = types.StepType.TEXT_RESPONSE
    elif step_dict.get("thinking"):
      step_type = types.StepType.THINKING

    source_str = step_dict.get("source")
    source = (
        _SOURCE_MAP.get(source_str, types.StepSource.UNKNOWN)
        if isinstance(source_str, str)
        else types.StepSource.UNKNOWN
    )

    status_str = step_dict.get("state")
    status = (
        _STATUS_MAP.get(status_str, types.StepStatus.UNKNOWN)
        if isinstance(status_str, str)
        else types.StepStatus.UNKNOWN
    )

    is_from_model = source == types.StepSource.MODEL
    is_done = status == types.StepStatus.DONE
    has_text = bool(step_dict.get("text"))
    is_target_user = step_dict.get("target") == "TARGET_USER"
    is_complete_response = (
        is_from_model and is_done and has_text and is_target_user
    )

    structured_output = None
    if step_type == types.StepType.FINISH:
      finish_dict = step_dict.get("finish", {})
      output_string = finish_dict.get("output_string")
      if output_string:
        try:
          structured_output = json.loads(output_string)
        except json.JSONDecodeError:
          logging.warning(
              "Failed to parse structured output JSON.", exc_info=True
          )

    error_field = step_dict.get("error", {})
    error_msg = error_field.get("error_message", "")
    http_code = error_field.get("http_code", 0)

    return cls(
        id=id_str,
        step_index=step_idx,
        trajectory_id=traj_id,
        parent_trajectory_id=parent_traj_id,
        depth=depth_val,
        type=step_type,
        source=source,
        status=status,
        content=step_dict.get("text", ""),
        content_delta=step_dict.get("text_delta", ""),
        thinking=step_dict.get("thinking", ""),
        thinking_delta=step_dict.get("thinking_delta", ""),
        tool_calls=tool_calls,
        error=error_msg,
        http_code=http_code,
        is_complete_response=is_complete_response,
        target=_TARGET_MAP.get(
            step_dict.get("target", ""), types.StepTarget.UNKNOWN
        ),
        structured_output=structured_output,
    )


class LocalHarnessEventProcessor:
  """Processes OutputEvent messages from the local harness and routes them."""

  def __init__(
      self,
      *,
      send_input_event_fn: Callable[
          [localharness_pb2.InputEvent], Coroutine[Any, Any, None]
      ],
      hook_runner: h_runner.HookRunner | None = None,
      tool_runner: t_runner.ToolRunner | None = None,
      dynamic_policy_map: dict[str, policy_lib.Policy] | None = None,
      initial_usage: types.UsageMetadata | None = None,
      initial_trajectory_usages: dict[str, types.UsageMetadata] | None = None,
  ):
    self._send_input_event = send_input_event_fn
    self._hook_runner = hook_runner
    self._tool_runner = tool_runner
    self._dynamic_policy_map: dict[str, policy_lib.Policy] = (
        dynamic_policy_map or {}
    )
    self.step_queue = asyncio.Queue()
    self.is_idle = asyncio.Event()
    self.is_idle.set()
    self.session_end_done = asyncio.Event()
    self.main_trajectory_id = None
    self._step_trackers: dict[tuple[str, int], _StepTracker] = {}
    self._background_tasks = set()
    self._hook_router = (
        HookRouter(hook_runner, self._send_input_event, _extract_tool_result)
        if hook_runner
        else None
    )
    self._cumulative_usage: types.UsageMetadata = (
        initial_usage.model_copy()
        if initial_usage is not None
        else types.UsageMetadata()
    )
    self._trajectory_usages: dict[str, types.UsageMetadata] = (
        initial_trajectory_usages.copy()
        if initial_trajectory_usages is not None
        else {}
    )
    self._turn_stop_reason: types.StopReason = types.StopReason.UNSPECIFIED

  def reset_for_turn(self) -> None:
    self.is_idle.clear()
    self.main_trajectory_id = None
    self._turn_stop_reason = types.StopReason.UNSPECIFIED
    while not self.step_queue.empty():
      try:
        self.step_queue.get_nowait()
      except asyncio.QueueEmpty:
        break

  @property
  def cumulative_usage(self) -> types.UsageMetadata:
    """Returns total cumulative token usage from the backend."""
    return self._cumulative_usage

  @property
  def trajectory_usages(self) -> dict[str, types.UsageMetadata]:
    """Returns per-trajectory cumulative token usage from the backend."""
    return self._trajectory_usages.copy()

  @property
  def _last_turn_stop_reason(self) -> types.StopReason:
    """Returns the stop reason of the most recent turn."""
    return self._turn_stop_reason

  async def cancel_background_tasks(self) -> None:
    for task in self._background_tasks:
      task.cancel()
    if self._background_tasks:
      await asyncio.gather(*self._background_tasks, return_exceptions=True)
    self._background_tasks.clear()

  def _get_turn_context(self) -> hooks.TurnContext:
    assert self._hook_runner is not None
    if self._hook_router and self._hook_router.current_turn_context:
      return self._hook_router.current_turn_context
    return hooks.TurnContext(self._hook_runner.session_context)

  def _run_in_background(self, coro) -> None:
    t = asyncio.create_task(coro)
    self._background_tasks.add(t)
    t.add_done_callback(self._background_tasks.discard)

  async def process_event(self, event: localharness_pb2.OutputEvent) -> None:
    """Processes OutputEvents from the harness, routes steps, and dispatches tools."""
    if event.HasField("policy_decision_request"):
      self._run_in_background(
          self._handle_policy_decision_request(event.policy_decision_request)
      )
      return

    if event.HasField("call_hook_request"):
      if self._hook_router:
        self._run_in_background(
            self._hook_router.handle(event.call_hook_request)
        )
      else:
        resp = localharness_pb2.CallHookResponse(
            request_id=event.call_hook_request.request_id,
            empty_result=localharness_pb2.EmptyResult(),
        )
        self._run_in_background(
            self._send_input_event(
                localharness_pb2.InputEvent(call_hook_response=resp)
            )
        )
      return

    if event.HasField("session_end_response"):
      self.session_end_done.set()
      return

    if event.HasField("step_update"):
      step_update = event.step_update

      # Update local step tracker state
      step_key = (step_update.trajectory_id, step_update.step_index)
      if step_key not in self._step_trackers:
        self._step_trackers[step_key] = _StepTracker()

      tracker = self._step_trackers[step_key]
      tracker.update_state(step_update.state)

      # Push step update to queue
      step_dict = json_format.MessageToDict(
          event.step_update, preserving_proto_field_name=True
      )
      parsed_step = LocalConnectionStep.from_dict(step_dict)
      step_obj = parsed_step

      step_obj_for_queue = step_obj
      if self._tool_runner and step_obj.tool_calls:
        is_local_custom_tool = False
        for tc in step_obj.tool_calls:
          if tc.name in self._tool_runner.tool_names:
            is_local_custom_tool = True
            break

        if is_local_custom_tool:
          # During live execution of a local custom tool, the Go harness
          # sends both a StepUpdate event with custom_tool and a websocket
          # tool_call event. To prevent duplicate ToolCallStart events on the
          # client, we suppress the tool calls from this StepUpdate before
          # putting it in the queue.
          #
          # This filtering is done here rather than in from_dict() because
          # during history resumption, from_dict() is called directly to load
          # historical steps. Since no websocket tool_call events are replayed
          # during resumption, from_dict() must parse the custom_tool from the
          # history payload to reconstruct the steps in the session.
          step_obj_for_queue = step_obj.model_copy(update={"tool_calls": []})

      await self.step_queue.put(step_obj_for_queue)

      # Record the main trajectory ID on the first step we see
      if self.main_trajectory_id is None and step_update.trajectory_id:
        self.main_trajectory_id = step_update.trajectory_id

      # Dispatch Telemetry Internal Step Hooks
      if (
          not tracker.pre_step_dispatched
          and self._hook_runner
          and step_update.state
          != localharness_pb2.StepUpdate.State.STATE_UNSPECIFIED
      ):
        tracker.pre_step_dispatched = True
        await self._hook_runner.dispatch_pre_step(
            self._get_turn_context(), step_obj
        )

      is_terminal = step_update.state in (
          localharness_pb2.StepUpdate.State.STATE_DONE,
          localharness_pb2.StepUpdate.State.STATE_ERROR,
      )
      if is_terminal and not tracker.post_step_dispatched and self._hook_runner:
        tracker.post_step_dispatched = True
        await self._hook_runner.dispatch_post_step(
            self._get_turn_context(), step_obj
        )

      # Process wait requests if this is a wait state
      if (
          step_update.state
          == localharness_pb2.StepUpdate.State.STATE_WAITING_FOR_USER
      ):
        if step_update.HasField("questions_request"):
          if tracker.mark_handled("questions_request"):
            self._run_in_background(self.handle_question_request(step_update))

        if step_update.HasField("tool_confirmation_request"):
          if tracker.mark_handled("tool_confirmation_request"):
            self._run_in_background(
                self.handle_tool_confirmation_request(step_update)
            )
      return

    if event.HasField("usage_update"):
      usage_update = event.usage_update
      if usage_update.HasField("total"):
        self._cumulative_usage = parse_usage_metadata(usage_update.total)
      for entry in usage_update.agents:
        if entry.trajectory_id and entry.HasField("usage"):
          self._trajectory_usages[entry.trajectory_id] = parse_usage_metadata(
              entry.usage
          )
      return

    if event.HasField("trajectory_state_update"):
      tsu = event.trajectory_state_update
      is_subagent = (
          self.main_trajectory_id
          and tsu.trajectory_id != self.main_trajectory_id
      )

      # Subagent execution is coordinated by the harness. The Python client
      # only tracks the main trajectory's idle state and ignores subagent
      # trajectory events (except for logging failures on exit).
      if is_subagent:
        if tsu.HasField("error"):
          logging.info("Subagent trajectory failed with error: %s", tsu.error)
        return

      if tsu.stop_reason:
        self._turn_stop_reason = _parse_stop_reason(tsu.stop_reason)

      if (
          tsu.state
          == localharness_pb2.TrajectoryStateUpdate.State.STATE_RUNNING
      ):
        if self.is_idle.is_set():
          self.is_idle.clear()

      elif (
          tsu.state
          == localharness_pb2.TrajectoryStateUpdate.State.STATE_FULLY_IDLE
      ):
        if tsu.HasField("error"):
          await self.step_queue.put(types.AntigravityExecutionError(tsu.error))
        self.is_idle.set()
        await self.step_queue.put(IDLE_SENTINEL)

      elif (
          tsu.state
          == localharness_pb2.TrajectoryStateUpdate.State.STATE_CANCELLED
      ):
        msg = tsu.error if tsu.HasField("error") else "Turn cancelled"
        await self.step_queue.put(types.AntigravityExecutionError(msg))
        self.is_idle.set()
        await self.step_queue.put(IDLE_SENTINEL)
      return

    if event.HasField("tool_call"):
      self._run_in_background(self.handle_tool_call(event.tool_call))
      return

  async def handle_question_request(
      self, step_update: localharness_pb2.StepUpdate
  ) -> None:
    """Handles question requests from the harness."""
    try:
      questions_list = []
      indices_to_hook = []
      for i, uq in enumerate(step_update.questions_request.questions):
        if uq.HasField("multiple_choice"):
          mc = uq.multiple_choice
          opts = [
              types.AskQuestionOption(id=str(j + 1), text=choice)
              for j, choice in enumerate(mc.choices)
          ]
          questions_list.append(
              types.AskQuestionEntry(question=mc.question, options=opts)
          )
          indices_to_hook.append(i)

      answers = [
          localharness_pb2.UserQuestionAnswer(unanswered=True)
          for _ in step_update.questions_request.questions
      ]

      if self._hook_runner and questions_list:
        ctx = self._get_turn_context()
        _, question_res, _ = await self._hook_runner.dispatch_interaction(
            turn_context=ctx,
            interaction_spec=types.AskQuestionInteractionSpec(
                questions=questions_list
            ),
        )
        if question_res:
          for orig_idx, r in zip(indices_to_hook, question_res.responses):
            ans = localharness_pb2.UserQuestionAnswer()
            if r.skipped:
              ans.unanswered = True
            else:
              mc_ans = localharness_pb2.MultipleChoiceAnswer()
              if r.selected_option_ids:
                indices = []
                for opt_id in r.selected_option_ids:
                  try:
                    indices.append(int(opt_id) - 1)
                  except ValueError:
                    pass
                mc_ans.selected_choice_indices[:] = indices
              if r.freeform_response:
                mc_ans.freeform_response = r.freeform_response
              ans.multiple_choice_answer.CopyFrom(mc_ans)
            answers[orig_idx] = ans
      elif not questions_list and step_update.questions_request.questions:
        logging.warning(
            "Received question_request with questions but none were"
            " multiple_choice. Skipping all."
        )
      elif not self._hook_runner:
        logging.warning(
            "Received question_request but no HookRunner is configured."
            " Skipping."
        )

      await self._send_question_response(step_update, answers)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("_handle_question_request failed; sending error")
      answers = [
          localharness_pb2.UserQuestionAnswer(
              multiple_choice_answer=localharness_pb2.MultipleChoiceAnswer(
                  freeform_response=f"SDK error processing question: {e!r}"
              )
          )
          for _ in step_update.questions_request.questions
      ]
      await self._send_question_response(step_update, answers)

  async def _send_question_response(
      self,
      step_update: localharness_pb2.StepUpdate,
      answers: list[localharness_pb2.UserQuestionAnswer],
  ) -> None:
    """Helper to format and send a UserQuestionsResponse."""
    resp = localharness_pb2.UserQuestionsResponse(
        trajectory_id=step_update.trajectory_id,
        step_index=step_update.step_index,
        response=localharness_pb2.UserQuestionsResponse.QuestionsResponse(
            answers=answers
        ),
    )
    input_event = localharness_pb2.InputEvent(question_response=resp)
    await self._send_input_event(input_event)

  async def handle_tool_confirmation_request(
      self, step_update: localharness_pb2.StepUpdate
  ) -> None:
    """Handles tool confirmation requests from the harness.

    Auto-accepts unconditionally. Pre-tool gating is handled by
    FirePreToolHook -> CallHookRequest -> HookRouter._handle_pre_tool.

    Args:
      step_update: The StepUpdate containing the tool confirmation request.
    """
    await self._send_tool_confirmation(step_update, accepted=True)

  async def _send_tool_confirmation(
      self, step_update: localharness_pb2.StepUpdate, accepted: bool
  ) -> None:
    """Helper to format and send a ToolConfirmation."""
    resp = localharness_pb2.ToolConfirmation(
        trajectory_id=step_update.trajectory_id,
        step_index=step_update.step_index,
        accepted=accepted,
    )
    input_event = localharness_pb2.InputEvent(tool_confirmation=resp)
    await self._send_input_event(input_event)

  async def handle_tool_call(
      self, tool_call: localharness_pb2.ToolCall
  ) -> None:
    """Handles tool execution and hook interception."""
    try:
      args = json.loads(tool_call.arguments_json or "{}")

      tc = types.ToolCall(id=tool_call.id, name=tool_call.name, args=args)

      tool_call_step = LocalConnectionStep(
          id=tool_call.id,
          step_index=1,
          type=types.StepType.TOOL_CALL,
          source=types.StepSource.MODEL,
          target=types.StepTarget.ENVIRONMENT,
          status=types.StepStatus.ACTIVE,
          tool_calls=[tc],
      )
      await self.step_queue.put(tool_call_step)

      if self._tool_runner:
        try:
          results = await self._tool_runner.process_tool_calls([tc])
          result = results[0]
        except Exception as e:  # pylint: disable=broad-except
          result = types.ToolResult(
              id=tool_call.id,
              name=tool_call.name,
              error=str(e),
              exception=e,
          )

        await self._send_tool_results([result])
      else:
        logging.warning(
            "Received tool call %s but no tool runner is configured. "
            "Yielding to user.",
            tool_call.name,
        )
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("_handle_tool_call failed; returning error to model")
      await self._send_tool_results([
          types.ToolResult(
              id=tool_call.id,
              name=tool_call.name,
              error=f"Internal SDK error: {e!r}",
          )
      ])

  def tool_result_to_dict(self, result: types.ToolResult) -> dict[str, Any]:
    """Converts a ToolResult to a dictionary representation."""
    if result.error is not None:
      return {"error": result.error}

    output = result.result
    if hasattr(output, "model_dump"):
      output = output.model_dump(mode="json")
    elif hasattr(output, "dict"):
      output = output.dict()

    try:
      output = _ANY_ADAPTER.dump_python(output, mode="json")
    except Exception:  # pylint: disable=broad-except
      logging.warning(
          "Pydantic serialization failed for tool result, falling back to"
          " string",
          exc_info=True,
      )
      output = str(output)

    if not isinstance(output, dict):
      return {"result": output}

    return output

  async def _send_tool_results(self, results: list[types.ToolResult]) -> None:
    """Sends tool execution results back to the harness."""
    for result in results:
      if not result.id:
        raise ValueError(
            f"ToolResult for '{result.name}' is missing an id. The"
            " LocalConnection protocol requires an id to correlate results"
            " with calls."
        )
      if result.error is not None:
        response = localharness_pb2.ToolResponse(
            id=result.id,
            error_message=result.error,
        )
      else:
        if struct_converter.has_proto_extensions(result.result):
          res_data = (
              result.result
              if struct_converter.is_structured(result.result)
              else {"result": result.result}
          )
          response = localharness_pb2.ToolResponse(
              id=result.id,
              response=struct_converter.to_struct(res_data),
              response_json=json.dumps(
                  struct_converter.to_json_fallback(res_data)
              ),
          )
        else:
          # Split any media out of the result so it reaches the model as
          # supplemental media instead of opaque base64 in response_json.
          cleaned_value, media = _extract_media_from_result(result.result)
          if media and cleaned_value is None:
            cleaned_value = f"Returned {len(media)} media attachment(s)."
          result_for_json = (
              result.model_copy(update={"result": cleaned_value})
              if media
              else result
          )
          response = localharness_pb2.ToolResponse(
              id=result.id,
              response_json=json.dumps(
                  self.tool_result_to_dict(result_for_json)
              ),
              supplemental_media=[
                  localharness_pb2.Media(
                      mime_type=item.mime_type,
                      data=item.data,
                      description=item.description,
                  )
                  for item in media
              ],
          )
      input_event = localharness_pb2.InputEvent(tool_response=response)
      await self._send_input_event(input_event)

  # ---------------------------------------------------------------------------
  # Dynamic policy evaluation
  # ---------------------------------------------------------------------------

  async def _handle_policy_decision_request(
      self, request: localharness_pb2.PolicyDecisionRequest
  ) -> None:
    """Evaluates a dynamic policy rule and sends the decision response."""
    rule_id = request.rule_id
    p = self._dynamic_policy_map.get(rule_id)

    if p is None:
      logging.error("Unknown policy rule_id: %s", rule_id)
      await self._send_policy_decision_response(
          request.request_id,
          outcome=localharness_pb2.POLICY_EVALUATION_OUTCOME_DENY,
          deny_reason=f"Unknown rule_id: {rule_id}",
      )
      return

    try:
      args_dict = json.loads(request.tool_args.arguments_json or "{}")
    except json.JSONDecodeError:
      args_dict = {}

    tool_call = types.ToolCall(
        name=request.tool_args.tool_name,
        args=args_dict,
        server_name=request.tool_args.server_name or None,
    )

    try:
      # Evaluate the `when` predicate if present.
      if p.when is not None:
        predicate_matched = await policy_lib._evaluate_predicate(p, tool_call)
        if not predicate_matched:
          await self._send_policy_decision_response(
              request.request_id,
              outcome=localharness_pb2.POLICY_EVALUATION_OUTCOME_NO_MATCH,
          )
          return

      # Apply the decision.
      if p.decision == policy_lib.Decision.ASK_USER:
        allow = await policy_lib._execute_ask_user(p, tool_call)
        await self._send_policy_decision_response(
            request.request_id,
            outcome=localharness_pb2.POLICY_EVALUATION_OUTCOME_ALLOW
            if allow
            else localharness_pb2.POLICY_EVALUATION_OUTCOME_DENY,
            deny_reason=""
            if allow
            else f"Denied by user ({p.name or p.tool}).",
        )
      elif p.decision == policy_lib.Decision.DENY:
        await self._send_policy_decision_response(
            request.request_id,
            outcome=localharness_pb2.POLICY_EVALUATION_OUTCOME_DENY,
            deny_reason=f"Denied by policy '{p.name or p.tool}'.",
        )
      elif p.decision == policy_lib.Decision.APPROVE:
        await self._send_policy_decision_response(
            request.request_id,
            outcome=localharness_pb2.POLICY_EVALUATION_OUTCOME_ALLOW,
        )
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Policy evaluation failed for rule_id=%s", rule_id)
      await self._send_policy_decision_response(
          request.request_id,
          outcome=localharness_pb2.POLICY_EVALUATION_OUTCOME_DENY,
          deny_reason=f"Policy evaluation error: {e}",
      )

  async def _send_policy_decision_response(
      self,
      request_id: str,
      *,
      outcome: "localharness_pb2.PolicyEvaluationOutcome",
      deny_reason: str = "",
  ) -> None:
    """Sends a PolicyDecisionResponse back to the localharness."""
    resp = localharness_pb2.PolicyDecisionResponse(
        request_id=request_id,
        outcome=outcome,
        deny_reason=deny_reason,
    )
    await self._send_input_event(
        localharness_pb2.InputEvent(policy_decision_response=resp)
    )
