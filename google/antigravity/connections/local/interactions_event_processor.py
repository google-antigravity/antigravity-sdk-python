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

"""Event processor for GAOS Interactions API events from localharness."""

import asyncio
import base64
import json
import logging
from typing import Any, Callable, Coroutine

from google.antigravity import types
from google.antigravity.connections import struct_converter
from google.antigravity.connections.local import event_processor
from google.antigravity.connections.local import interactions_config_converter
from google.antigravity.connections.local import interactions_step_assembler
from google.antigravity.connections.local.local_connection_config import normalize_wire_path
from google.antigravity.connections.local.local_connection_config import PROTO_FIELD_TO_SDK_NAME
from google.antigravity.connections.local.local_connection_config import WIRE_PATH_ARGUMENT_KEYS
from google.antigravity.hooks import hook_runner as h_runner
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy as policy_lib
from google.antigravity.tools import tool_runner as t_runner

_STOP_REASON_JSON_MAP: dict[str, types.StopReason] = {
    "max_model_calls_exceeded": types.StopReason.MAX_MODEL_CALLS_EXCEEDED,
    "max_tool_calls_exceeded": types.StopReason.MAX_TOOL_CALLS_EXCEEDED,
    "max_input_tokens_exceeded": types.StopReason.MAX_INPUT_TOKENS_EXCEEDED,
    "max_output_tokens_exceeded": types.StopReason.MAX_OUTPUT_TOKENS_EXCEEDED,
    "max_total_tokens_exceeded": types.StopReason.MAX_TOTAL_TOKENS_EXCEEDED,
    "quota_exhausted": types.StopReason.QUOTA_EXHAUSTED,
}


def _content_items_to_content(items: list[Any]) -> types.Content:
  """Converts a list of GAOS Content / UserInputContent dicts into SDK Content."""
  content_list: list[types.ContentPrimitive] = []
  for item in items:
    if not isinstance(item, dict):
      continue
    item_type = item.get("type")
    if item_type == "text":
      content_list.append(str(item.get("text", "")))
    elif item_type == "slash_command":
      try:
        content_list.append(
            types.SlashCommand(
                name=types.BuiltinSlashCommandName(str(item.get("name", "")))
            )
        )
      except ValueError:
        pass
    elif item_type in ("image", "audio", "video", "document"):
      raw_b64 = item.get("data", "")
      data_bytes = (
          base64.b64decode(raw_b64) if isinstance(raw_b64, str) else b""
      )
      try:
        content_list.append(
            types.from_bytes(
                data=data_bytes,
                mime_type=str(item.get("mime_type", "")),
            )
        )
      except ValueError:
        pass
  if not content_list:
    return ""
  if len(content_list) == 1:
    return content_list[0]
  return content_list


def _user_input_step_dict_to_content(ui_dict: dict[str, Any]) -> types.Content:
  """Converts a GAOS UserInputStep dict directly into SDK types.Content."""
  input_content = ui_dict.get("input_content")
  if isinstance(input_content, list) and input_content:
    return _content_items_to_content(input_content)
  content = ui_dict.get("content")
  if isinstance(content, str):
    return content
  if isinstance(content, dict):
    if content.get("type") == "text":
      return str(content.get("text", ""))
    return ""
  if isinstance(content, list):
    return _content_items_to_content(content)
  return ""


class InteractionsEventProcessor(event_processor.BaseLocalEventProcessor):
  """Processes GAOS Interactions ServerEvent dictionaries from localharness."""

  def __init__(
      self,
      *,
      send_json_event_fn: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
      root_interaction_id: str = "",
      hook_runner: h_runner.HookRunner | None = None,
      tool_runner: t_runner.ToolRunner | None = None,
      dynamic_policy_map: dict[str, policy_lib.Policy] | None = None,
      initial_usage: types.UsageMetadata | None = None,
      initial_trajectory_usages: dict[str, types.UsageMetadata] | None = None,
  ):
    self._send_json_event = send_json_event_fn
    super().__init__(
        hook_runner=hook_runner,
        tool_runner=tool_runner,
        dynamic_policy_map=dynamic_policy_map,
        initial_usage=initial_usage,
        initial_trajectory_usages=initial_trajectory_usages,
    )
    if root_interaction_id:
      self.main_trajectory_id = root_interaction_id
    self._step_assembler = (
        interactions_step_assembler.InteractionsStepAssembler(
            root_interaction_id=root_interaction_id
        )
    )
    self._pending_elicitations: dict[
        str, list[interactions_step_assembler.ParsedElicitation]
    ] = {}
    self._scheduled_elicitation_groups: set[str] = set()

  def reset_for_turn(self) -> None:
    root_id = self._step_assembler.root_interaction_id
    super().reset_for_turn()
    if root_id:
      self.main_trajectory_id = root_id

  async def process_event(self, event: dict[str, Any]) -> None:
    """Processes a single decoded GAOS ExternalServerEvent JSON dictionary."""
    event_type = str(event.get("event_type", ""))

    if event_type == "interaction.created":
      interaction_id = str(event.get("interaction_id", ""))
      if interaction_id:
        self._step_assembler.set_root_interaction_id(interaction_id)
        if self.main_trajectory_id is None:
          self.main_trajectory_id = interaction_id
      return

    if event_type == "interaction.completed":
      self.session_end_done.set()
      return

    if event_type == "call_hook_request":
      await self._handle_call_hook_request(event)
      return

    if event_type == "function_invocation":
      self._run_in_background(self.handle_function_invocation(event))
      return

    if event_type == "step.start":
      res = self._step_assembler.handle_step_start(event)
      if res.step is not None:
        await self._emit_step(
            res.step,
            dispatch_pre=res.dispatch_pre,
            dispatch_post=res.dispatch_post,
        )
      if res.elicitation is not None:
        self._enqueue_elicitation(res.elicitation)
      return

    if event_type == "step.delta":
      res = self._step_assembler.handle_step_delta(event)
      if res.step is not None:
        await self._emit_step(
            res.step,
            dispatch_pre=res.dispatch_pre,
            dispatch_post=res.dispatch_post,
        )
      return

    if event_type == "step.stop":
      res = self._step_assembler.handle_step_stop(event)
      if res.step_usage is not None:
        self._record_step_usage(res.step_usage, res.trajectory_id)
      if res.step is not None:
        await self._emit_step(
            res.step,
            dispatch_pre=res.dispatch_pre,
            dispatch_post=res.dispatch_post,
        )
      return

    if event_type == "usage_update":
      usage_payload = (
          event["usage_update"]
          if isinstance(event.get("usage_update"), dict)
          else event
      )
      step_usage = interactions_step_assembler.parse_interaction_usage(
          usage_payload
      )
      if step_usage is not None:
        sub_id = str(event.get("sub_interaction_id") or "")
        self._record_step_usage(step_usage, sub_id)
      return

    if event_type == "state_update":
      await self._handle_state_update(event)
      return

  def _record_step_usage(
      self, step_usage: types.UsageMetadata, trajectory_id: str = ""
  ) -> None:
    """Accumulates per-step UsageMetadata into cumulative and trajectory totals."""
    self._cumulative_usage = self._cumulative_usage + step_usage
    traj_id = (
        trajectory_id
        or self._step_assembler.root_interaction_id
        or self.main_trajectory_id
        or ""
    )
    if traj_id:
      prev = self._trajectory_usages.get(traj_id, types.UsageMetadata())
      self._trajectory_usages[traj_id] = prev + step_usage

  def _build_compaction_step(
      self, event: dict[str, Any]
  ) -> event_processor.LocalConnectionStep:
    """Builds a COMPACTION step from a GAOS call_hook_request(on_compaction) event."""
    oca = event.get("on_compaction_args")
    if not isinstance(oca, dict):
      oca = {}
    traj_id = str(oca.get("interaction_id") or oca.get("trajectory_id") or "")
    step_idx = int(oca.get("step_index", 0))
    summary = str(oca.get("summary", ""))
    return event_processor.LocalConnectionStep(
        id=f"{traj_id}:{step_idx}" if traj_id else str(step_idx),
        step_index=step_idx,
        trajectory_id=traj_id,
        type=types.StepType.COMPACTION,
        source=types.StepSource.SYSTEM,
        target=types.StepTarget.USER,
        status=types.StepStatus.DONE,
        content=summary or "Context compaction",
    )

  async def _handle_call_hook_request(self, event: dict[str, Any]) -> None:
    """Handles a GAOS call_hook_request event directly against HookRunner."""
    self._run_in_background(self._dispatch_hook_request(event))

  async def _dispatch_hook_request(self, event: dict[str, Any]) -> None:
    """Dispatches a GAOS call_hook_request dict to HookRunner and sends call_hook_response."""
    request_id = str(event.get("request_id", ""))
    hook_type = str(event.get("type", ""))
    hook_name = str(event.get("name", ""))
    if not self._hook_runner:
      await self._send_json_event({
          "event_type": "call_hook_response",
          "request_id": request_id,
          "empty_result": {},
      })
      return

    resp: dict[str, Any] = {
        "event_type": "call_hook_response",
        "request_id": request_id,
    }
    try:
      if hook_type == "on_session_start":
        await self._hook_runner.dispatch_session_start()
        resp["empty_result"] = {}
      elif hook_type == "on_session_end":
        await self._hook_runner.dispatch_session_end()
        resp["empty_result"] = {}
      elif hook_type == "pre_turn":
        user_input: types.Content = ""
        pta = event.get("pre_turn_args")
        if isinstance(pta, dict):
          ui_dict = pta.get("user_input")
          if isinstance(ui_dict, dict):
            user_input = _user_input_step_dict_to_content(ui_dict)
        res, turn_ctx = await self._hook_runner.dispatch_pre_turn(user_input)
        self._current_turn_context = turn_ctx
        if res is None or res.allow:
          resp["pre_turn_result"] = {"decision": "allow"}
        else:
          ptr: dict[str, Any] = {"decision": "deny"}
          if res.message:
            ptr["reason"] = res.message
          resp["pre_turn_result"] = ptr
      elif hook_type == "post_turn":
        response_text = ""
        pta = event.get("post_turn_args")
        if isinstance(pta, dict):
          response_text = str(pta.get("response_text", ""))
        turn_ctx = self._get_turn_context()
        await self._hook_runner.dispatch_post_turn(turn_ctx, response_text)
        self._current_turn_context = None
        resp["empty_result"] = {}
      elif hook_type == "pre_tool":
        await self._handle_pre_tool_hook(event, resp)
      elif hook_type == "post_tool":
        await self._handle_post_tool_hook(event, resp)
      elif hook_type == "on_tool_error":
        await self._handle_on_tool_error_hook(event, resp)
      elif hook_type == "on_compaction":
        await self._handle_on_compaction_hook(event, resp)
      elif hook_type == "stop":
        await self._handle_stop_hook(event, resp)
      else:
        logging.warning(
            "Unknown or unhandled hook received -> type: %s, name: %s",
            hook_type,
            hook_name,
        )
        resp["empty_result"] = {}
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Hook %s failed", hook_name)
      resp["error_message"] = f"Hook failed: {e!r}"

    await self._send_json_event(resp)

  async def _handle_pre_tool_hook(
      self, event: dict[str, Any], resp: dict[str, Any]
  ) -> None:
    """Dispatches a pre_tool hook directly from GAOS JSON."""
    assert self._hook_runner is not None
    tool_name = ""
    args: dict[str, Any] = {}
    server_name: str | None = None
    call_id: str | None = None
    step_id: str | None = None
    pta = event.get("pre_tool_args")
    if isinstance(pta, dict):
      raw_tool_name = str(pta.get("tool_name", ""))
      tool_name = PROTO_FIELD_TO_SDK_NAME.get(raw_tool_name, raw_tool_name)
      raw_args_json = pta.get("arguments_json")
      if isinstance(raw_args_json, str) and raw_args_json:
        args = json.loads(raw_args_json)
      if pta.get("server_name"):
        server_name = str(pta["server_name"])
      if pta.get("call_id"):
        call_id = str(pta["call_id"])
        step_id = call_id
      for key in WIRE_PATH_ARGUMENT_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val:
          args[key] = normalize_wire_path(val)

    canonical_path: str | None = None
    for key in WIRE_PATH_ARGUMENT_KEYS:
      val = args.get(key)
      if isinstance(val, str) and val:
        canonical_path = val
        break

    tc = types.ToolCall(
        name=tool_name,
        args=args,
        id=call_id,
        step_id=step_id,
        server_name=server_name,
        canonical_path=canonical_path,
    )
    turn_ctx = self._get_turn_context()
    result, _, _ = await self._hook_runner.dispatch_pre_tool_call(
        turn_context=turn_ctx, tool_call=tc
    )
    if result.allow:
      ptr: dict[str, Any] = {"decision": "allow"}
      if result.modified_args is not None:
        ptr["modified_args"] = result.modified_args
      resp["pre_tool_result"] = ptr
    else:
      ptr = {"decision": "deny"}
      if result.message:
        ptr["reason"] = result.message
      resp["pre_tool_result"] = ptr

  async def _handle_post_tool_hook(
      self, event: dict[str, Any], resp: dict[str, Any]
  ) -> None:
    """Dispatches a post_tool hook directly from GAOS JSON."""
    assert self._hook_runner is not None
    tool_name = ""
    server_name: str | None = None
    call_id: str | None = None
    step_id: str | None = None
    result_val: Any = None
    pta = event.get("post_tool_args")
    if isinstance(pta, dict):
      raw_tool_name = str(pta.get("tool_name", ""))
      tool_name = PROTO_FIELD_TO_SDK_NAME.get(raw_tool_name, raw_tool_name)
      server_name = str(pta["server_name"]) if pta.get("server_name") else None
      if pta.get("call_id"):
        call_id = str(pta["call_id"])
        step_id = call_id
      raw_result = str(pta.get("result", ""))
      result_val = raw_result
      if raw_result:
        extracted = event_processor._extract_tool_result(  # pylint: disable=protected-access
            tool_name, raw_result
        )
        if extracted is not None:
          result_val = extracted

    tool_result = types.ToolResult(
        name=tool_name,
        id=call_id,
        step_id=step_id,
        server_name=server_name,
        result=result_val,
    )
    turn_ctx = self._get_turn_context()
    op_ctx = hooks.OperationContext(turn_ctx)
    await self._hook_runner.dispatch_post_tool_call(op_ctx, tool_result)
    resp["empty_result"] = {}

  async def _handle_on_tool_error_hook(
      self, event: dict[str, Any], resp: dict[str, Any]
  ) -> None:
    """Dispatches an on_tool_error hook directly from GAOS JSON."""
    assert self._hook_runner is not None
    error_message = "Tool failed"
    tool_name = ""
    server_name: str | None = None
    call_id: str | None = None
    step_id: str | None = None
    ote = event.get("on_tool_error_args")
    if isinstance(ote, dict):
      error_message = str(ote.get("error_message") or error_message)
      raw_tool_name = str(ote.get("tool_name", ""))
      tool_name = PROTO_FIELD_TO_SDK_NAME.get(raw_tool_name, raw_tool_name)
      server_name = str(ote["server_name"]) if ote.get("server_name") else None
      if ote.get("call_id"):
        call_id = str(ote["call_id"])
        step_id = call_id

    error = types.ToolExecutionError(
        error_message, tool_name, server_name, call_id=call_id, step_id=step_id
    )
    turn_ctx = self._get_turn_context()
    op_ctx = hooks.OperationContext(turn_ctx)
    hook_result, recovery_val = await self._hook_runner.dispatch_on_tool_error(
        op_ctx, error
    )
    if (
        hook_result.allow
        and isinstance(recovery_val, str)
        and recovery_val.strip()
    ):
      resp["on_tool_error_result"] = {
          "custom_error_message": recovery_val.strip()
      }
    else:
      resp["empty_result"] = {}

  async def _handle_on_compaction_hook(
      self, event: dict[str, Any], resp: dict[str, Any]
  ) -> None:
    """Dispatches an on_compaction hook directly from GAOS JSON."""
    assert self._hook_runner is not None
    step_obj = self._build_compaction_step(event)
    turn_ctx = self._get_turn_context()
    await self._hook_runner.dispatch_compaction(turn_ctx, step_obj)
    resp["empty_result"] = {}

  async def _handle_stop_hook(
      self, event: dict[str, Any], resp: dict[str, Any]
  ) -> None:
    """Dispatches a stop hook directly from GAOS JSON."""
    assert self._hook_runner is not None
    stop_args = types.StopArgs()
    sa = event.get("stop_args")
    if isinstance(sa, dict):
      stop_reason_str = str(sa.get("stop_reason", ""))
      stop_args = types.StopArgs(
          response_text=str(sa.get("response_text", "")),
          trajectory_id=str(
              sa.get("interaction_id") or sa.get("trajectory_id") or ""
          ),
          continuation_count=int(sa.get("continuation_count", 0)),
          stop_reason=_STOP_REASON_JSON_MAP.get(
              stop_reason_str, types.StopReason.UNSPECIFIED
          ),
          error_message=str(sa.get("error_message", "")),
      )
    turn_ctx = self._get_turn_context()
    result = await self._hook_runner.dispatch_stop(turn_ctx, stop_args)
    if result.decision == types.StopDecision.CONTINUE:
      self._current_turn_context = turn_ctx
      sr: dict[str, Any] = {"decision": "continue"}
      if result.reason and result.reason.strip():
        sr["reason"] = result.reason.strip()
      resp["stop_result"] = sr
    else:
      self._current_turn_context = None
      resp["stop_result"] = {"decision": "allow_stop"}

  async def _handle_state_update(self, event: dict[str, Any]) -> None:
    """Processes a GAOS state_update event."""
    sub_id = str(event.get("sub_interaction_id") or "")
    parent_id = str(event.get("parent_interaction_id") or "")
    if sub_id and parent_id:
      self._step_assembler.record_parent_trajectory(sub_id, parent_id)

    root_id = (
        self._step_assembler.root_interaction_id or self.main_trajectory_id
    )
    is_subagent = bool(sub_id and root_id and sub_id != root_id) or bool(
        sub_id and not root_id and parent_id
    )
    if is_subagent:
      if event.get("error"):
        logging.info(
            "Subagent trajectory failed with error: %s", event["error"]
        )
      return

    reason_str = str(event.get("reason") or "")
    if reason_str in _STOP_REASON_JSON_MAP:
      self._turn_stop_reason = _STOP_REASON_JSON_MAP[reason_str]

    state_str = str(event.get("state") or "")
    if state_str in ("running", "waiting_for_tasks"):
      if self.is_idle.is_set():
        self.is_idle.clear()
    elif state_str == "fully_idle":
      if event.get("error"):
        await self.step_queue.put(
            types.AntigravityExecutionError(str(event["error"]))
        )
      self.is_idle.set()
      await self.step_queue.put(event_processor.IDLE_SENTINEL)
    elif state_str == "cancelled":
      msg = str(event.get("error") or "Turn cancelled")
      await self.step_queue.put(types.AntigravityExecutionError(msg))
      self.is_idle.set()
      await self.step_queue.put(event_processor.IDLE_SENTINEL)

  def _enqueue_elicitation(
      self, elicitation: interactions_step_assembler.ParsedElicitation
  ) -> None:
    """Buffers elicitations belonging to the same question batch and schedules dispatch."""
    group_key = elicitation.group_key
    self._pending_elicitations.setdefault(group_key, []).append(elicitation)
    if group_key not in self._scheduled_elicitation_groups:
      self._scheduled_elicitation_groups.add(group_key)
      self._run_in_background(self._flush_elicitation_group(group_key))

  async def _flush_elicitation_group(self, group_key: str) -> None:
    """Yields briefly so multi-question elicitation_call steps batch together, then dispatches."""
    await asyncio.sleep(0)
    self._scheduled_elicitation_groups.discard(group_key)
    batch = self._pending_elicitations.pop(group_key, [])
    if not batch:
      return

    elicitation_ids = [item.elicitation_id for item in batch]
    questions_list = [item.question for item in batch]
    try:
      question_res = await self._dispatch_question_interaction(questions_list)
      if question_res is None:
        events = interactions_config_converter.question_responses_to_elicitation_result_events(
            elicitation_ids,
            responses=[
                types.QuestionResponse(skipped=True) for _ in elicitation_ids
            ],
        )
      elif question_res.cancelled:
        events = interactions_config_converter.question_responses_to_elicitation_result_events(
            elicitation_ids,
            cancelled=True,
        )
      else:
        events = interactions_config_converter.question_responses_to_elicitation_result_events(
            elicitation_ids,
            responses=question_res.responses,
        )
      for ev in events:
        await self._send_json_event(ev)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("_flush_elicitation_group failed; sending error answer")
      err_events = interactions_config_converter.question_responses_to_elicitation_result_events(
          elicitation_ids,
          responses=[
              types.QuestionResponse(
                  freeform_response=f"SDK error processing question: {e!r}"
              )
              for _ in elicitation_ids
          ],
      )
      for ev in err_events:
        await self._send_json_event(ev)

  async def handle_function_invocation(self, event: dict[str, Any]) -> None:
    """Executes a client-side function tool invocation and sends a function_result step."""
    call_id = str(event.get("call_id") or event.get("id") or "")
    tool_name = str(event.get("name") or "")
    traj_id = str(
        event.get("sub_interaction_id")
        or self._step_assembler.root_interaction_id
        or self.main_trajectory_id
        or ""
    )
    raw_args = event.get("arguments")
    args = dict(raw_args) if isinstance(raw_args, dict) else {}

    try:
      tc = types.ToolCall(id=call_id, name=tool_name, args=args)
      tool_call_step = event_processor.LocalConnectionStep(
          id=call_id,
          step_index=1,
          trajectory_id=traj_id,
          type=types.StepType.TOOL_CALL,
          source=types.StepSource.MODEL,
          target=types.StepTarget.ENVIRONMENT,
          status=types.StepStatus.ACTIVE,
          tool_calls=[tc],
      )
      await self.step_queue.put(tool_call_step)

      result = await self._execute_client_tool(tc)
      if result is not None:
        await self._send_tool_results([result])
    except Exception as e:  # pylint: disable=broad-except
      logging.exception(
          "handle_function_invocation failed; returning error to model"
      )
      await self._send_tool_results([
          types.ToolResult(
              id=call_id,
              name=tool_name,
              error=f"Internal SDK error: {e!r}",
          )
      ])

  async def _send_tool_results(self, results: list[types.ToolResult]) -> None:
    """Sends tool execution results back to the harness as GAOS function_result steps."""
    for result in results:
      if not result.id:
        raise ValueError(
            f"ToolResult for '{result.name}' is missing an id. The"
            " InteractionsConnection protocol requires an id to correlate"
            " results with calls."
        )
      tool_name = (
          result.name.value
          if isinstance(result.name, types.BuiltinTools)
          else str(result.name)
      )
      if result.error is not None:
        ev = interactions_config_converter.tool_result_to_function_result_event(
            call_id=result.id,
            tool_name=tool_name,
            error_message=result.error,
        )
      else:
        sanitized_output, _ = event_processor._extract_media_from_result(  # pylint: disable=protected-access
            result.result
        )
        sanitized_result = result.model_copy(
            update={"result": sanitized_output}
        )
        if event_processor.has_proto_extensions(sanitized_result.result):
          res_data = (
              sanitized_result.result
              if struct_converter.is_structured(sanitized_result.result)
              else {"result": sanitized_result.result}
          )
          fallback = struct_converter.to_json_fallback(res_data)
          result_dict = (
              fallback if isinstance(fallback, dict) else {"result": fallback}
          )
        else:
          result_dict = self.tool_result_to_dict(sanitized_result)
        ev = interactions_config_converter.tool_result_to_function_result_event(
            call_id=result.id,
            tool_name=tool_name,
            result_dict=result_dict,
        )
      await self._send_json_event(ev)
