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

"""Local connection for the Google Antigravity SDK."""

import asyncio
import collections
import importlib.metadata
import importlib.resources
import inspect
import json
import logging
import os
import pathlib
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
from typing import Any, AsyncIterator, Callable, Sequence, cast

from google.genai import types as genai_types
from google.protobuf import json_format
from typing_extensions import override
import websockets

from google.antigravity.proto import localharness_pb2
from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.connections.local import event_processor
from google.antigravity.connections.local import local_connection_config
from google.antigravity.hooks import hook_runner as h_runner
from google.antigravity.hooks import policy
from google.antigravity.tools import schema_utils
from google.antigravity.tools import tool_runner as t_runner

LocalConnectionStep = event_processor.LocalConnectionStep
IDLE_SENTINEL = event_processor.IDLE_SENTINEL
CLOSE_SENTINEL = event_processor.CLOSE_SENTINEL
# In some cases e.g. during eval runs, the harness needs extra time to
# shutdown cleanly. So we give it ample time. This constant is needed to make
# tests fast.
_PROCESS_WAIT_TIMEOUT_SECONDS = 3 * 60
_MAX_WEBSOCKET_CONNECT_RETRIES = 5


_SESSION_CONTINUATION_MODE_MAP = {
    types.SessionContinuationMode.RESUME: localharness_pb2.HarnessConfig.RESUME,
    types.SessionContinuationMode.CREATE_OR_RESUME: (
        localharness_pb2.HarnessConfig.CREATE_OR_RESUME
    ),
    types.SessionContinuationMode.CREATE_ONLY: (
        localharness_pb2.HarnessConfig.CREATE_ONLY
    ),
}


def to_proto_session_continuation_mode(
    mode: types.SessionContinuationMode | None,
) -> localharness_pb2.HarnessConfig.SessionContinuationMode:
  if mode in _SESSION_CONTINUATION_MODE_MAP:
    return _SESSION_CONTINUATION_MODE_MAP[mode]
  return localharness_pb2.HarnessConfig.SESSION_CONTINUATION_MODE_UNSPECIFIED


_AGENT_BEHAVIOR_MAP = {
    types.AgentBehavior.AUTONOMOUS: localharness_pb2.AGENT_BEHAVIOR_AUTONOMOUS,
    types.AgentBehavior.INTERACTIVE: (
        localharness_pb2.AGENT_BEHAVIOR_INTERACTIVE
    ),
    types.AgentBehavior.MINIMAL: localharness_pb2.AGENT_BEHAVIOR_MINIMAL,
}


def to_proto_agent_behavior(
    behavior: types.AgentBehavior | None,
) -> localharness_pb2.AgentBehavior:
  if behavior is not None and behavior in _AGENT_BEHAVIOR_MAP:
    return _AGENT_BEHAVIOR_MAP[behavior]
  return localharness_pb2.AGENT_BEHAVIOR_AUTONOMOUS


def to_proto_model_type(
    model_type: types.ModelType,
) -> localharness_pb2.ModelType:
  if model_type == types.ModelType.TEXT:
    return localharness_pb2.MODEL_TYPE_TEXT
  if model_type == types.ModelType.IMAGE:
    return localharness_pb2.MODEL_TYPE_IMAGE
  return localharness_pb2.MODEL_TYPE_UNSPECIFIED


def build_gemini_options_proto(
    options: types.GeminiModelOptions | None,
) -> localharness_pb2.GeminiModelOptions:
  """Converts SDK GeminiModelOptions to localharness protobuf representation."""
  proto = localharness_pb2.GeminiModelOptions()
  if options:
    if options.thinking_level:
      proto.thinking_level = str(options.thinking_level.value)
    if options.service_tier:
      if hasattr(options.service_tier, "value"):
        proto.service_tier = str(options.service_tier.value)
      else:
        proto.service_tier = str(options.service_tier)
  return proto


def build_retry_config_proto(
    config: types.RetryConfig | None,
) -> localharness_pb2.RetryConfig | None:
  """Builds a RetryConfig proto from a RetryConfig model."""
  if not config:
    return None
  proto = localharness_pb2.RetryConfig()
  if config.api_retry:
    api_data = config.api_retry.model_dump(exclude_none=True)
    if api_data:
      proto.api_retry.CopyFrom(
          localharness_pb2.ModelAPIRetryConfig(**api_data)
      )
  if config.model_output_retry:
    output_data = config.model_output_retry.model_dump(exclude_none=True)
    if output_data:
      proto.model_output_retry.CopyFrom(
          localharness_pb2.ModelOutputRetryConfig(**output_data)
      )
  if (
      not proto.HasField("api_retry")
      and not proto.HasField("model_output_retry")
  ):
    return None
  return proto


def build_budget_config_proto(
    config: types.BudgetConfig | None,
) -> localharness_pb2.BudgetConfig | None:
  """Builds a BudgetConfig proto from a BudgetConfig model."""
  if not config:
    return None
  data = config.model_dump(exclude_none=True)
  has_limits = any(
      k in data
      for k in (
          "max_model_calls",
          "max_tool_calls",
          "max_input_tokens",
          "max_output_tokens",
          "max_total_tokens",
      )
  )
  if not has_limits:
    return None
  if "scope" in data:
    scope_val = getattr(data["scope"], "value", data["scope"])
    if isinstance(scope_val, int):
      data["scope"] = scope_val
    elif isinstance(scope_val, str):
      scope_name = (
          scope_val
          if scope_val.startswith("BUDGET_SCOPE_")
          else f"BUDGET_SCOPE_{scope_val}"
      )
      data["scope"] = (
          localharness_pb2.BudgetConfig.BudgetScope.Value(scope_name)
      )
  return localharness_pb2.BudgetConfig(**data)


def build_models_proto(
    models: list[types.ModelTarget],
) -> list[localharness_pb2.ModelConfig]:
  """Builds a list of ModelConfig protos from a ModelConfig list."""
  protos = []
  for m in models:
    proto = localharness_pb2.ModelConfig(
        name=m.name or "",
        types=[to_proto_model_type(t) for t in m.types],
    )
    if isinstance(m.endpoint, types.GeminiAPIEndpoint):
      api_endpoint_proto = localharness_pb2.GeminiAPIEndpoint(
          base_url=m.endpoint.base_url or "",
          http_headers=m.endpoint.http_headers or {},
          api_key=m.endpoint.api_key or "",
      )
      if m.endpoint.options and m.endpoint.options.model_dump(
          exclude_none=True
      ):
        api_endpoint_proto.options.CopyFrom(
            build_gemini_options_proto(m.endpoint.options)
        )
      proto.gemini_api_endpoint.CopyFrom(api_endpoint_proto)
    elif isinstance(m.endpoint, types.VertexEndpoint):
      vertex_endpoint_proto = localharness_pb2.VertexEndpoint(
          base_url=m.endpoint.base_url or "",
          http_headers=m.endpoint.http_headers or {},
          project=m.endpoint.project or "",
          location=m.endpoint.location or "",
          api_key=m.endpoint.api_key or "",
      )
      if m.endpoint.options and m.endpoint.options.model_dump(
          exclude_none=True
      ):
        vertex_endpoint_proto.options.CopyFrom(
            build_gemini_options_proto(m.endpoint.options)
        )
      proto.vertex_endpoint.CopyFrom(vertex_endpoint_proto)
    else:
      raise ValueError(f"Unrecognized endpoint type: {type(m.endpoint)}")
    protos.append(proto)
  return protos


def callable_to_tool_proto(
    fn: Callable[..., Any],
    tool_runner: t_runner.ToolRunner | None = None,
) -> localharness_pb2.Tool:
  """Converts a Python callable to a localharness Tool proto.

  Uses google.genai.types.FunctionDeclaration for schema extraction.
  If a ``tool_runner`` is provided, the runner's ``get_public_callable``
  is used to strip injectable parameters (e.g. ``ToolContext``) from
  the schema so the model never sees them.

  Args:
      fn: The Python callable to convert.
      tool_runner: Optional ToolRunner that owns schema-hiding logic.

  Returns:
      A localharness_pb2.Tool proto.
  """
  if isinstance(fn, t_runner.ToolWithSchema):
    return localharness_pb2.Tool(
        name=getattr(fn, "__name__", ""),
        description=fn.__doc__ or "",
        parameters_json_schema=json.dumps(
            schema_utils.normalize_schema(fn.input_schema)
        ),
    )

  # Use the ToolRunner's public callable to strip injectable params.
  target_fn = fn
  tool_name = getattr(fn, "__name__", None) or type(fn).__name__
  if tool_runner is not None and tool_name in tool_runner.tools:
    target_fn = tool_runner.get_public_callable(tool_name)

  if not hasattr(target_fn, "__name__"):
    orig_fn = target_fn

    def wrapped(*args, **kwargs):
      return orig_fn(*args, **kwargs)

    wrapped.__name__ = tool_name
    setattr(wrapped, "__doc__", getattr(orig_fn, "__doc__", None))
    try:
      setattr(wrapped, "__signature__", inspect.signature(orig_fn))
    except (ValueError, TypeError):
      setattr(
          wrapped, "__annotations__", getattr(orig_fn, "__annotations__", {})
      )
    target_fn = wrapped

  decl = genai_types.FunctionDeclaration.from_callable_with_api_option(
      callable=target_fn,
      api_option="GEMINI_API",
  )
  if decl.parameters:
    parameters = decl.parameters.model_dump(exclude_none=True)
  elif decl.parameters_json_schema:
    parameters = decl.parameters_json_schema
  else:
    parameters = {"type": "object", "properties": {}}
  return localharness_pb2.Tool(
      name=decl.name,
      description=decl.description or "",
      parameters_json_schema=json.dumps(
          schema_utils.normalize_schema(parameters)
      ),
  )


# Strip null bytes and replace dangerous control characters before protobuf
# serialization so C-extension string bindings do not raise or truncate.
# Note: Canonical sanitization is enforced at the Go localharness ingress layer.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitize_prompt(text: str) -> str:
  """Strips null bytes and replaces dangerous control characters with spaces."""
  if not text:
    return ""
  sanitized = _CONTROL_CHAR_RE.sub(" ", text)
  if not sanitized.strip():
    return " "
  return sanitized


def _get_ws_close_code(e: websockets.ConnectionClosed) -> int | str | None:
  """Safely retrieves the WebSocket close code across websockets library versions."""
  rcvd = getattr(e, "rcvd", None)
  if rcvd is not None and hasattr(rcvd, "code"):
    return rcvd.code
  sent = getattr(e, "sent", None)
  if sent is not None and hasattr(sent, "code"):
    return sent.code
  if rcvd is None and sent is None:
    return 1006
  return getattr(e, "code", None)


class LocalConnection(connection.Connection):
  """Connection to the Go-based local harness."""

  def __init__(
      self,
      process: subprocess.Popen[bytes] | None,
      ws: Any,
      tool_runner: t_runner.ToolRunner | None = None,
      hook_runner: h_runner.HookRunner | None = None,
      initial_history: Sequence[types.Step] | None = None,
      env: dict[str, str] | None = None,
      debug_config: connection.DebugConfig | None = None,
      dynamic_policy_map: dict[str, "policy.Policy"] | None = None,
      initial_usage: types.UsageMetadata | None = None,
      initial_trajectory_usages: dict[str, types.UsageMetadata] | None = None,
  ):
    self._hook_runner = hook_runner
    self._process = process
    self._ws = ws
    self._tool_runner = tool_runner
    self._env = env
    self._debug_config = debug_config
    self.__initial_history = initial_history or []
    self._client_cancelled = False
    self._is_receiving = False

    # Flag set early in disconnect() so the reader loop can distinguish
    # expected closures from harness crashes.
    self._disconnecting = False

    self._processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=self._send_input_event,
        hook_runner=hook_runner,
        tool_runner=tool_runner,
        dynamic_policy_map=dynamic_policy_map,
        initial_usage=initial_usage,
        initial_trajectory_usages=initial_trajectory_usages,
    )

    self._reader_task = asyncio.create_task(self._ws_reader_loop())

    # Stderr lines from the Go harness, captured by a background thread.
    # Retained in a bounded deque so the reader loop can surface harness
    # error messages when the WebSocket closes unexpectedly.
    self._stderr_lines: collections.deque[str] = collections.deque(maxlen=100)
    self._stderr_thread: threading.Thread | None = None

  @property
  def is_idle(self) -> bool:
    """Returns True if the connection is idle and ready for input."""
    return self._processor.is_idle.is_set()

  @property
  def _initial_history(self) -> Sequence[types.Step]:
    """Returns the pre-existing session steps restored during handshake."""
    return self.__initial_history

  @property
  @override
  def debug_config(self) -> connection.DebugConfig | None:
    """Returns the debug configuration for this connection."""
    return self._debug_config

  @property
  def conversation_id(self) -> str:
    """Returns the conversation identifier, if one exists."""
    return self._processor.main_trajectory_id or ""

  @property
  def cumulative_usage(self) -> types.UsageMetadata:
    """Returns total cumulative token usage from the backend."""
    return self._processor.cumulative_usage

  @property
  def trajectory_usages(self) -> dict[str, types.UsageMetadata]:
    """Returns per-trajectory cumulative token usage from the backend."""
    return self._processor.trajectory_usages

  @property
  def _last_turn_stop_reason(self) -> types.StopReason:
    """Returns the stop reason of the most recent turn from the backend."""
    return self._processor._last_turn_stop_reason  # pylint: disable=protected-access

  async def send(self, prompt: types.Content | None, **kwargs: Any) -> None:
    """Sends a prompt to the agent.

    Args:
      prompt: The user prompt or content to send.
      **kwargs: Strategy-specific options.
    """
    self._client_cancelled = False
    self._processor.reset_for_turn()

    if prompt is None:
      content_list = [""]
    elif isinstance(prompt, collections.abc.Sequence) and not isinstance(
        prompt, (str, bytes)
    ):
      content_list = prompt
    else:
      content_list = [prompt]
    user_input_pb = localharness_pb2.UserInput(
        parts=[to_proto_input_content(c) for c in content_list]
    )
    event = localharness_pb2.InputEvent(user_input=user_input_pb)

    await self._send_input_event(event)

  async def receive_steps(
      self,
  ) -> AsyncIterator[event_processor.LocalConnectionStep]:
    """Receives steps as they complete from the agent."""
    if self._is_receiving:
      raise RuntimeError(
          "Concurrent receive_steps() calls are not supported on this"
          " connection."
      )
    self._is_receiving = True
    try:
      if self.is_idle and self._processor.step_queue.empty():
        if self._client_cancelled:
          raise types.AntigravityCancelledError()
        return

      while True:
        if self.is_idle and self._processor.step_queue.empty():
          if self._client_cancelled:
            raise types.AntigravityCancelledError()
          return

        step_obj = await self._processor.step_queue.get()

        if isinstance(step_obj, Exception):
          raise step_obj

        if step_obj is event_processor.IDLE_SENTINEL:
          continue
        if step_obj is None:
          if self._client_cancelled:
            raise types.AntigravityCancelledError()
          return
        if isinstance(step_obj, Exception):
          raise step_obj

        step_obj = cast(event_processor.LocalConnectionStep, step_obj)
        yield step_obj

        # Detect platform-level errors (source=SYSTEM) and propagate them.
        if (
            step_obj.status == types.StepStatus.ERROR
            and step_obj.source == types.StepSource.SYSTEM
        ):
          http_code = getattr(step_obj, "http_code", 0)
          if http_code in (400, 401, 403):
            raise types.AntigravityConnectionError(
                step_obj.error or "System error occurred."
            )
          else:
            logging.warning(
                "System step error (HTTP %s): %s", http_code, step_obj.error
            )
    finally:
      self._is_receiving = False

  async def wait_for_idle(self) -> None:
    """Blocks until the connection becomes idle."""
    await self._processor.is_idle.wait()
    while not self._processor.step_queue.empty():
      try:
        self._processor.step_queue.get_nowait()
      except asyncio.QueueEmpty:
        break

  def _start_stderr_reader(self, stderr_stream) -> None:
    """Starts a background daemon thread that drains the harness stderr.

    Args:
      stderr_stream: The binary stderr stream from the harness process.
    """

    def _drain():
      try:
        for raw_line in stderr_stream:
          line = raw_line.decode("utf-8", errors="replace").rstrip()
          self._stderr_lines.append(line)
          logging.info("harness stderr: %s", line)
      except ValueError:
        pass  # Stream closed.

    t = threading.Thread(target=_drain, daemon=True, name="harness-stderr")
    t.start()
    self._stderr_thread = t

  async def disconnect(self) -> None:
    """Tears down the harness connection in a careful order."""
    self._disconnecting = True
    hook_error = None

    # Dispatch session end hook before tearing down via Go localharness RPC.
    if self._hook_runner and self._hook_runner.on_session_end_hooks:
      try:
        await self._send_input_event(
            localharness_pb2.InputEvent(session_end_request=True)
        )
        await self._processor.session_end_done.wait()
      except Exception as e:  # pylint: disable=broad-except
        hook_error = e

    try:
      # Cancel and await background tasks in the processor
      await self._processor.cancel_background_tasks()

      self._reader_task.cancel()
      try:
        await self._reader_task
      except asyncio.CancelledError:
        pass

      # Close the WebSocket first.
      try:
        await asyncio.wait_for(self._ws.close(), timeout=0.5)
      except asyncio.TimeoutError:
        pass

      # Close stdin to signal the Go main loop to exit.
      if self._process and self._process.stdin:
        self._process.stdin.close()

      # Wait for the process to exit, escalating if needed.
      if self._process:
        try:
          self._process.wait(timeout=_PROCESS_WAIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
          self._process.terminate()
          try:
            self._process.wait(timeout=1)
          except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=1)
    finally:
      if hook_error is not None:
        raise hook_error

  async def cancel(self) -> None:
    """Cancels the current turn."""
    self._client_cancelled = True
    event = localharness_pb2.InputEvent(halt_request=True)
    await self._send_input_event(event)

  async def _ws_reader_loop(self) -> None:
    """Reads OutputEvents from the WebSocket and delegates to processor."""
    try:
      async for raw_msg in self._ws:
        logging.debug("RAW WS MSG: %s", raw_msg)
        event = localharness_pb2.OutputEvent()
        json_format.Parse(raw_msg, event)
        await self._processor.process_event(event)
    except websockets.ConnectionClosed as e:
      close_code = _get_ws_close_code(e)
      if self._disconnecting:
        # Expected closure.
        logging.info("WebSocket closed (code %s); normal shutdown.", close_code)
      else:
        # Unexpected closure.
        stderr_tail = "\n".join(self._stderr_lines) or "(no stderr output)"
        error_msg = (
            f"Harness process exited unexpectedly (WS close code {close_code})."
            f"\nHarness stderr:\n{stderr_tail}"
        )
        logging.error(error_msg)
        await self._processor.step_queue.put(
            types.AntigravityConnectionError(error_msg)
        )

    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Error in reader loop: %s", e)
      await self._processor.step_queue.put(
          types.AntigravityConnectionError(f"Error in reader loop: {e}")
      )
    finally:
      await self._processor.step_queue.put(event_processor.CLOSE_SENTINEL)

  async def _send_input_event(self, event: localharness_pb2.InputEvent) -> None:
    """Helper to send an InputEvent over the WebSocket."""
    await self._ws.send(json_format.MessageToJson(event))

  async def send_trigger_notification(self, content: str) -> None:
    """Sends a trigger message to the agent."""
    event = localharness_pb2.InputEvent(automated_trigger=content)
    await self._send_input_event(event)

  # Testing proxies to preserve compatibility with unit tests
  @property
  def _step_queue(self) -> asyncio.Queue[Any]:
    return self._processor.step_queue

  @property
  def _is_idle(self) -> asyncio.Event:
    return self._processor.is_idle

  @property
  def _main_trajectory_id(self) -> str | None:
    return self._processor.main_trajectory_id

  @_main_trajectory_id.setter
  def _main_trajectory_id(self, val: str | None) -> None:
    self._processor.main_trajectory_id = val

  async def _handle_tool_call(
      self, tool_call: localharness_pb2.ToolCall
  ) -> None:
    """Handles tool execution and hook interception."""
    await self._processor.handle_tool_call(tool_call)

  def _tool_result_to_dict(self, result: types.ToolResult) -> dict[str, Any]:
    """Converts a ToolResult to a dictionary representation."""
    return self._processor.tool_result_to_dict(result)

  async def _handle_question_request(
      self, step_update: localharness_pb2.StepUpdate
  ) -> None:
    """Handles question requests from the harness."""
    await self._processor.handle_question_request(step_update)

  async def _handle_tool_confirmation_request(
      self, step_update: localharness_pb2.StepUpdate
  ) -> None:
    """Handles tool confirmation requests from the harness."""
    await self._processor.handle_tool_confirmation_request(step_update)


def to_proto_input_content(
    content: types.ContentPrimitive,
) -> localharness_pb2.UserInput.Part:
  """Converts dynamic prompt fragments into proto Parts."""
  if isinstance(content, str):
    return localharness_pb2.UserInput.Part(text=_sanitize_prompt(content))

  if isinstance(content, types.SlashCommand):
    sc_pb = localharness_pb2.UserInput.SlashCommand(
        name=content.name,
    )
    return localharness_pb2.UserInput.Part(slash_command=sc_pb)

  is_semantic_media = isinstance(
      content, (types.Image, types.Document, types.Audio, types.Video)
  )
  if is_semantic_media:
    media_pb = localharness_pb2.UserInput.Media(
        mime_type=content.mime_type,
        data=content.data,
        description=content.description,
    )
    return localharness_pb2.UserInput.Part(media=media_pb)

  raise TypeError(f"Unsupported prompt content type: {type(content)}")


def _get_sdk_version() -> str:
  """Returns the version of the Google Antigravity SDK."""
  try:
    return importlib.metadata.version("google-antigravity")
  except importlib.metadata.PackageNotFoundError:
    # Default to a development version if package metadata is not found.
    return "0.0.0-dev"


_HARNESS_PATH_ENV_VAR = "ANTIGRAVITY_HARNESS_PATH"


def _get_default_binary_path_external(env: dict[str, str] | None) -> str:
  """Returns the default localharness binary path."""
  # 1. Check passed value in env first.
  if env and _HARNESS_PATH_ENV_VAR in env:
    return env[_HARNESS_PATH_ENV_VAR]

  # 2. Check variable in os.environ.
  if harness_path := os.environ.get(_HARNESS_PATH_ENV_VAR):
    return harness_path

  # 3. Try importlib.metadata (Robust wheel discovery)
  # This is immune to sys.path shadowing by a local repository directory.
  try:
    dist = importlib.metadata.distribution("google-antigravity")
    if dist.files:
      for f in dist.files:
        normalized_path = str(f).replace("\\", "/")
        if normalized_path.endswith((
            "google/antigravity/bin/localharness",
            "google/antigravity/bin/localharness.exe",
        )):
          binary_path = os.path.abspath(str(f.locate()))
          if os.path.exists(binary_path):
            return binary_path
  except (importlib.metadata.PackageNotFoundError, ValueError, AttributeError):
    pass

  # 4. Try importlib.resources (External Wheel fallback)
  try:
    # Using 'google.antigravity' as the package name.
    # This assumes the binary is located at google/antigravity/bin/localharness
    # in the installed package.
    suffix = (
        "bin/localharness.exe"
        if sys.platform == "win32"
        else "bin/localharness"
    )
    binary_path = str(
        importlib.resources.files("google.antigravity").joinpath(suffix)
    )
    if os.path.exists(binary_path):
      return binary_path
  except (ImportError, AttributeError, KeyError):
    pass

  # 5. Fallback: Check if it's in the system PATH
  if path := shutil.which("localharness"):
    return path

  raise RuntimeError(
      "Could not find default localharness binary. "
      "Please specify binary_path explicitly, set the "
      f"{_HARNESS_PATH_ENV_VAR} environment variable, or ensure it is in your "
      "PATH. Note: If you are running from the root of the repository, the "
      "local source tree might shadow your pip-installed package and prevent "
      "resource discovery."
  )


_get_default_binary_path = _get_default_binary_path_external


def _to_mcp_server_proto(
    server_cfg: types.McpServerConfig,
) -> localharness_pb2.McpServerConfig:
  """Converts an McpServerConfig to a McpServerConfig proto."""

  if isinstance(server_cfg, types.McpStdioServer):
    return localharness_pb2.McpServerConfig(
        name=server_cfg.name,
        enabled_tools=server_cfg.enabled_tools or [],
        disabled_tools=server_cfg.disabled_tools or [],
        timeout_seconds=server_cfg.timeout_seconds or 0,
        stdio=localharness_pb2.McpStdioTransport(
            command=server_cfg.command,
            args=server_cfg.args,
            env=server_cfg.env or {},
        ),
    )
  elif isinstance(server_cfg, types.McpStreamableHttpServer):
    return localharness_pb2.McpServerConfig(
        name=server_cfg.name,
        enabled_tools=server_cfg.enabled_tools or [],
        disabled_tools=server_cfg.disabled_tools or [],
        timeout_seconds=server_cfg.timeout_seconds or 0,
        http=localharness_pb2.McpHttpTransport(
            url=server_cfg.url,
            headers=server_cfg.headers or {},
        ),
    )
  raise ValueError(f"Unknown McpServerConfig type: {type(server_cfg)}")


def to_system_instructions_proto(
    instructions: str | types.SystemInstructions | None,
) -> localharness_pb2.SystemInstructions | None:
  """Converts string or SystemInstructions model to localharness SystemInstructions proto.

  String instructions and TemplatedSystemInstructions map to
  AppendedSystemInstructions (appending to default agent system instructions).
  CustomSystemInstructions map to CustomSystemInstructions (replacing default
  system instructions).

  Args:
    instructions: System instructions as string or SystemInstructions object.

  Returns:
    The localharness SystemInstructions proto, or None if instructions is empty.
  """
  if not instructions:
    return None
  if isinstance(instructions, str):
    instructions = types.TemplatedSystemInstructions(
        sections=[types.SystemInstructionSection(content=instructions)]
    )

  proto = localharness_pb2.SystemInstructions()
  if isinstance(instructions, types.CustomSystemInstructions):
    proto.custom.CopyFrom(
        localharness_pb2.CustomSystemInstructions(
            part=[
                localharness_pb2.CustomSystemInstructions.Part(
                    text=instructions.text
                )
            ]
        )
    )
  elif isinstance(instructions, types.TemplatedSystemInstructions):
    appended = localharness_pb2.AppendedSystemInstructions()
    if instructions.identity:
      appended.custom_identity = instructions.identity
    for sec in instructions.sections:
      appended.appended_sections.add(title=sec.title, content=sec.content)
    proto.appended.CopyFrom(appended)
  return proto


class LocalConnectionStrategy(connection.ConnectionStrategy):
  """Strategy for establishing a LocalConnection."""

  _models: list[types.ModelTarget]
  _system_instructions: types.SystemInstructions | None
  _connection: LocalConnection | None

  def __init__(
      self,
      *,
      tool_runner: t_runner.ToolRunner | None = None,
      hook_runner: h_runner.HookRunner | None = None,
      models: list[types.ModelTarget] | None = None,
      skills_paths: list[str] | None = None,
      system_instructions: str | types.SystemInstructions | None = None,
      capabilities_config: types.CapabilitiesConfig | None = None,
      conversation_id: str | None = None,
      session_continuation_mode: types.SessionContinuationMode | None = None,
      save_dir: str | None = None,
      workspaces: list[str] | None = None,
      app_data_dir: str | None = None,
      mcp_servers: Sequence[types.McpServerConfig] | None = None,
      env: dict[str, str] | None = None,
      subagents: list[types.SubagentConfig] | None = None,
      debug_config: connection.DebugConfig | None = None,
      retry_config: types.RetryConfig | None = None,
      budget_config: types.BudgetConfig | None = None,
      policies: list[policy.Policy] | None = None,
      tools: Sequence[Callable[..., Any] | str] | None = None,
  ):
    """Initializes the instance.

    Args:
      tool_runner: Optional ToolRunner for custom tools.
      hook_runner: Optional HookRunner for custom hooks.
      models: Optional list of model targets.
      skills_paths: Optional list of paths to search for skills.
      system_instructions: Optional SystemInstructions or string shorthand.
      capabilities_config: Optional CapabilitiesConfig to configure tools.
      conversation_id: Optional conversation identifier.
      session_continuation_mode: Optional mode for establishing a connection.
      save_dir: Optional directory to save trajectories.
      workspaces: Optional list of workspace paths.
      app_data_dir: Optional directory for harness app data.
      mcp_servers: Optional sequence of MCP server configurations.
      env: Optional dictionary of custom environment variables.
      subagents: Optional list of static subagent configurations.
      debug_config: Optional debug configuration for the connection.
      retry_config: Optional retry configuration for model API and outputs.
      budget_config: Optional session budget configuration.
      policies: Optional list of policy rules for the Go evaluator.
      tools: Optional list of tools for the root agent.
    """
    self._binary_path = _get_default_binary_path(env)
    self._tool_runner = tool_runner
    self._hook_runner = hook_runner
    self._tools = tools
    self._connection: LocalConnection | None = None
    self._mcp_servers = mcp_servers or []
    self._models: list[types.ModelTarget] = models or []
    self._skills_paths = skills_paths
    self._env = env
    self._debug_config = debug_config
    self._retry_config = retry_config
    self._budget_config = budget_config
    self._policies = policies or []
    # Maps rule_id -> Policy for dynamic rules evaluated during tool execution.
    self._dynamic_policy_map: dict[str, policy.Policy] = {}

    # Normalize str shorthand to SystemInstructions model.
    self._system_instructions: types.SystemInstructions | None = None
    if isinstance(system_instructions, str):
      self._system_instructions = types.TemplatedSystemInstructions(
          sections=[types.SystemInstructionSection(content=system_instructions)]
      )
    else:
      self._system_instructions = system_instructions
    self._capabilities_config = (
        capabilities_config or types.CapabilitiesConfig()
    )
    self._conversation_id = conversation_id
    self._session_continuation_mode = session_continuation_mode
    self._save_dir = save_dir
    self._workspaces = local_connection_config.normalize_workspace_paths(
        workspaces
    )
    self._app_data_dir = app_data_dir
    self._subagents = subagents or []

  def _resolve_active_tools(
      self,
      cfg: types.CapabilitiesConfig | types.SubagentCapabilities | None,
      is_subagent: bool = False,
  ) -> set[types.BuiltinTools]:
    if cfg is None:
      if is_subagent:
        cfg = types.SubagentCapabilities(
            enabled_tools=types.BuiltinTools.read_only()
        )
      else:
        cfg = types.CapabilitiesConfig(
            enabled_tools=types.BuiltinTools.read_only(),
            enable_subagents=False,
        )
    all_tools = set(types.BuiltinTools)
    if cfg.enabled_tools is not None:
      return set(cfg.enabled_tools)
    if cfg.disabled_tools is not None:
      return all_tools - set(cfg.disabled_tools)
    return all_tools

  def _to_system_instructions_proto(
      self,
      instructions: str | types.SystemInstructions | None,
  ) -> localharness_pb2.SystemInstructions | None:
    return to_system_instructions_proto(instructions)

  def _to_subagent_system_instructions_proto(
      self,
      instructions: str | types.SystemInstructions | None,
  ) -> localharness_pb2.SystemInstructions | None:
    return to_system_instructions_proto(instructions)

  def _to_harness_side_tools_proto(
      self,
      cfg: types.CapabilitiesConfig | types.SubagentCapabilities | None,
      is_subagent: bool = False,
  ) -> localharness_pb2.HarnessSideTools:
    active_tools = self._resolve_active_tools(cfg, is_subagent=is_subagent)
    subagent_enabled = False
    max_depth = None
    allowed_subagents = []

    if cfg is not None:
      subagent_enabled = getattr(cfg, "enable_subagents", True) and (
          types.BuiltinTools.START_SUBAGENT in active_tools
      )
      max_depth = getattr(cfg, "max_subagent_depth", None)
      allowed_subagents = cfg.allowed_subagents or []
    elif not is_subagent:
      subagent_enabled = types.BuiltinTools.START_SUBAGENT in active_tools

    subagents_proto = localharness_pb2.SubagentsConfig(
        enabled=subagent_enabled,
        allowed_subagents=allowed_subagents,
    )
    if max_depth is not None:
      subagents_proto.max_nesting_depth = max_depth

    run_cmd_cfg = None
    if cfg is not None:
      run_cmd_cfg = getattr(cfg, "run_command_config", None) or getattr(
          cfg, "run_command", None
      )
    enable_daemon = (
        run_cmd_cfg.enable_daemons if run_cmd_cfg is not None else False
    )
    timeout_ms = (
        int(round(run_cmd_cfg.timeout_seconds * 1000))
        if run_cmd_cfg is not None and run_cmd_cfg.timeout_seconds is not None
        else 0
    )
    enable_sandbox = (
        run_cmd_cfg.enable_sandbox if run_cmd_cfg is not None else False
    )

    return localharness_pb2.HarnessSideTools(
        subagents=subagents_proto,
        find=localharness_pb2.FindToolConfig(
            enabled=types.BuiltinTools.FIND_FILE in active_tools
        ),
        user_questions=localharness_pb2.UserQuestionsConfig(
            enabled=types.BuiltinTools.ASK_QUESTION in active_tools
        ),
        run_command=localharness_pb2.RunCommandToolConfig(
            enabled=types.BuiltinTools.RUN_COMMAND in active_tools,
            enable_daemon_commands=enable_daemon,
            max_timeout_ms=timeout_ms,
            enable_sandbox=enable_sandbox,
        ),
        file_edit=localharness_pb2.FileEditToolConfig(
            enabled=types.BuiltinTools.EDIT_FILE in active_tools
        ),
        view_file=localharness_pb2.ViewFileToolConfig(
            enabled=types.BuiltinTools.VIEW_FILE in active_tools
        ),
        write_to_file=localharness_pb2.WriteToFileToolConfig(
            enabled=types.BuiltinTools.CREATE_FILE in active_tools
        ),
        grep_search=localharness_pb2.GrepSearchToolConfig(
            enabled=types.BuiltinTools.SEARCH_DIR in active_tools
        ),
        list_dir=localharness_pb2.ListDirToolConfig(
            enabled=types.BuiltinTools.LIST_DIR in active_tools
        ),
        generate_image=localharness_pb2.GenerateImageToolConfig(
            enabled=types.BuiltinTools.GENERATE_IMAGE in active_tools,
        ),
        search_web=localharness_pb2.SearchWebToolConfig(
            enabled=types.BuiltinTools.SEARCH_WEB in active_tools
        ),
        read_url_content=localharness_pb2.ReadUrlContentToolConfig(
            enabled=types.BuiltinTools.READ_URL_CONTENT in active_tools
        ),
    )

  def _build_custom_subagents_protos(
      self,
      all_tool_protos: dict[str, localharness_pb2.Tool],
  ) -> list[localharness_pb2.CustomAgent]:
    """Resolves and builds CustomAgent configuration protos for subagents."""
    custom_agents_protos = []
    for subagent in self._subagents:
      capabilities = subagent.capabilities or types.SubagentCapabilities(
          enabled_tools=types.BuiltinTools.read_only(),
      )

      resolved_subagent_tools = []
      for tool in subagent.tools or []:
        if isinstance(tool, str):
          name = tool
          if name in all_tool_protos:
            resolved_subagent_tools.append(all_tool_protos[name])
          else:
            resolved_subagent_tools.append(localharness_pb2.Tool(name=name))
        elif callable(tool):
          proto = callable_to_tool_proto(tool, tool_runner=self._tool_runner)
          all_tool_protos[proto.name] = proto
          resolved_subagent_tools.append(proto)
        else:
          raise ValueError(
              f"Invalid tool type in subagent '{subagent.name}' tools list:"
              f" {tool}"
          )

      custom_agents_protos.append(
          localharness_pb2.CustomAgent(
              name=subagent.name,
              description=subagent.description,
              system_instructions=self._to_subagent_system_instructions_proto(
                  subagent.system_instructions
              ),
              harness_side_tools=self._to_harness_side_tools_proto(
                  capabilities, is_subagent=True
              ),
              tools=resolved_subagent_tools,
              agent_behavior=to_proto_agent_behavior(
                  capabilities.agent_behavior
              ),
          )
      )
    return custom_agents_protos

  def _build_harness_config(self) -> localharness_pb2.HarnessConfig:
    """Translates Pydantic config objects into a HarnessConfig proto."""
    all_tool_protos = {}
    if self._tool_runner:
      for fn in self._tool_runner.tools.values():
        proto = callable_to_tool_proto(fn, tool_runner=self._tool_runner)
        all_tool_protos[proto.name] = proto

    root_tool_protos = []
    if self._tools is not None:
      for tool in self._tools:
        if isinstance(tool, str):
          if tool in all_tool_protos:
            root_tool_protos.append(all_tool_protos[tool])
          else:
            root_tool_protos.append(localharness_pb2.Tool(name=tool))
        elif callable(tool):
          proto = callable_to_tool_proto(tool, tool_runner=self._tool_runner)
          all_tool_protos[proto.name] = proto
          root_tool_protos.append(proto)
    elif self._tool_runner:
      # Fallback when _tools is not explicitly specified: exclude tools
      # exclusive to subagents.
      subagent_tool_names = set()
      for sa in self._subagents:
        for t in sa.tools or []:
          if isinstance(t, str):
            subagent_tool_names.add(t)
          elif callable(t):
            subagent_proto = callable_to_tool_proto(
                t, tool_runner=self._tool_runner
            )
            subagent_tool_names.add(subagent_proto.name)
      root_tool_protos = [
          proto
          for name, proto in all_tool_protos.items()
          if name not in subagent_tool_names
      ]

    system_instructions_proto = self._to_system_instructions_proto(
        self._system_instructions
    )

    models_protos = []
    if self._models:
      models_protos = build_models_proto(self._models)
    workspace_protos = [
        localharness_pb2.Workspace(
            filesystem_workspace=localharness_pb2.FilesystemWorkspace(
                directory=pathlib.Path(p).as_posix()
            )
        )
        for p in self._workspaces
    ]

    harness_side_tools = self._to_harness_side_tools_proto(
        self._capabilities_config
    )

    mcp_server_protos = [
        _to_mcp_server_proto(s) for s in self._mcp_servers or []
    ]

    enabled_hooks = self._get_enabled_hooks()

    compaction_config = None
    if self._capabilities_config.compaction_threshold is not None:
      compaction_config = localharness_pb2.CompactionConfig(
          checkpoint_interval_tokens=self._capabilities_config.compaction_threshold
      )

    custom_agents_protos = self._build_custom_subagents_protos(all_tool_protos)

    harness_config = localharness_pb2.HarnessConfig(
        tools=root_tool_protos,
        system_instructions=system_instructions_proto,
        cascade_id=self._conversation_id or "",
        session_continuation_mode=to_proto_session_continuation_mode(
            self._session_continuation_mode
        ),
        models=models_protos,
        workspaces=workspace_protos,
        skills_paths=self._skills_paths or [],
        harness_side_tools=harness_side_tools,
        # 0 tells the harness to use its default.
        compaction_threshold=(
            self._capabilities_config.compaction_threshold or 0
        ),
        compaction_config=compaction_config,
        finish_tool_schema_json=(
            self._capabilities_config.finish_tool_schema_json or ""
        ),
        app_data_dir=self._app_data_dir
        or str(
            (pathlib.Path("~") / ".gemini" / "antigravity")
            .expanduser()
            .resolve()
        ),
        mcp_servers=mcp_server_protos,
        enabled_hooks=enabled_hooks,
        custom_subagents=custom_agents_protos,
        agent_behavior=to_proto_agent_behavior(
            self._capabilities_config.agent_behavior
        ),
    )
    if self._retry_config:
      retry_proto = build_retry_config_proto(self._retry_config)
      if retry_proto:
        harness_config.retry_config.CopyFrom(retry_proto)

    if self._budget_config:
      budget_proto = build_budget_config_proto(self._budget_config)
      if budget_proto:
        harness_config.budget_config.CopyFrom(budget_proto)

    if self._policies:
      policy_config, self._dynamic_policy_map = policy._to_policy_config_proto(
          self._policies
      )
      harness_config.policy_config.CopyFrom(policy_config)

    return harness_config

  def _get_enabled_hooks(self) -> list[Any]:
    """Returns a list of proto enum IDs for registered hook collections."""
    enabled_hooks = []
    if not self._hook_runner:
      return enabled_hooks

    hook_mapping = [
        (
            self._hook_runner.on_session_start_hooks,
            localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_START,
        ),
        (
            self._hook_runner.on_session_end_hooks,
            localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_END,
        ),
        (
            self._hook_runner.pre_turn_hooks,
            localharness_pb2.LIFECYCLE_HOOK_PRE_TURN,
        ),
        (
            self._hook_runner.post_turn_hooks,
            localharness_pb2.LIFECYCLE_HOOK_POST_TURN,
        ),
        (
            self._hook_runner.pre_tool_call_decide_hooks,
            localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL,
        ),
        (
            self._hook_runner.post_tool_call_hooks,
            localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
        ),
        (
            self._hook_runner.on_tool_error_hooks,
            localharness_pb2.LIFECYCLE_HOOK_ON_TOOL_ERROR,
        ),
        (
            self._hook_runner.on_compaction_hooks,
            localharness_pb2.LIFECYCLE_HOOK_ON_COMPACTION,
        ),
        (
            self._hook_runner.stop_hooks,
            localharness_pb2.LIFECYCLE_HOOK_STOP,
        ),
    ]
    for hooks_list, hook_type in hook_mapping:
      if hooks_list:
        enabled_hooks.append(hook_type)
    return enabled_hooks

  @property
  @override
  def debug_config(self) -> connection.DebugConfig | None:
    """Returns the debug configuration for this strategy."""
    return self._debug_config

  def connect(self) -> connection.Connection:
    """Returns the established Connection."""
    if not hasattr(self, "_connection") or self._connection is None:
      raise RuntimeError(
          "Connection not established. Use as a context manager."
      )
    return self._connection

  def _validate_connection(self) -> None:
    """Validates that all required configurations are present."""
    if not self._models:
      return  # Backend handles default model selection.

    for m in self._models:
      if m.endpoint:
        try:
          m.endpoint.validate_endpoint()
        except ValueError as e:
          raise types.AntigravityValidationError(str(e)) from e
      else:
        raise types.AntigravityValidationError(
            f"Model '{m.name}' must have an endpoint configured."
        )

  async def _connect_websocket(
      self, port: int, api_key: str, process: subprocess.Popen[bytes]
  ) -> tuple[Any, str]:
    """Attempts to connect to the local harness WebSocket with backoff."""
    for attempt in range(_MAX_WEBSOCKET_CONNECT_RETRIES):
      for host in ("localhost", "127.0.0.1"):
        ws_url = f"ws://{host}:{port}/"
        try:
          ws = await websockets.connect(
              ws_url,
              additional_headers={"x-goog-api-key": api_key},
              max_size=None,
          )
          return ws, ws_url
        except (OSError, websockets.WebSocketException) as e:
          last_exception = e
      await asyncio.sleep(0.1 * (2**attempt))

    process.kill()
    stderr = ""
    if process.stderr is not None:
      stderr = process.stderr.read().decode("utf-8")
    raise RuntimeError(
        f"Failed to connect to WebSocket at {ws_url} after"
        f" {_MAX_WEBSOCKET_CONNECT_RETRIES} attempts. Stderr: {stderr}"
    ) from last_exception

  async def __aenter__(self) -> None:
    """Starts the backend."""
    self._validate_connection()

    harness_config = self._build_harness_config()
    sdk_version = _get_sdk_version()
    client_info_proto = localharness_pb2.ClientInfo(
        language="python",
        version=sdk_version,
        language_version=platform.python_version(),
        os=platform.system().lower(),
        os_version=platform.release(),
    )
    env_map = (
        {str(k): str(v) for k, v in self._env.items()} if self._env else {}
    )
    input_config = localharness_pb2.InputConfig(
        storage_directory=self._save_dir or "",
        client_info=client_info_proto,
        env=env_map,
    )

    merged_env = {**os.environ, **env_map} if self._env is not None else None

    process = subprocess.Popen(
        [self._binary_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=merged_env,
    )

    serialized = input_config.SerializeToString()
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    # Note for humans: Pack length as 4-byte uint (little-endian)
    process.stdin.write(struct.pack("<I", len(serialized)) + serialized)
    process.stdin.flush()
    raw_len = process.stdout.read(4)
    if not raw_len:
      stderr_output = process.stderr.read().decode("utf-8")
      raise RuntimeError(
          f"Failed to read length from stdout. Stderr: {stderr_output}"
      )
    length = struct.unpack("<I", raw_len)[0]
    output_config = localharness_pb2.OutputConfig()
    output_config.ParseFromString(process.stdout.read(length))
    # Retry the WebSocket connection with backoff. The harness process may
    # need a moment to start listening after writing its OutputConfig. We try
    # localhost first and fall back to 127.0.0.1, as some environments may not
    # resolve localhost.
    ws, ws_url = await self._connect_websocket(
        output_config.port, output_config.api_key, process
    )

    try:
      init_event = localharness_pb2.InitializeConversationEvent(
          config=harness_config
      )
      await ws.send(json_format.MessageToJson(init_event))
      raw_init_resp = await ws.recv()
      initial_history: list[types.Step] = []
      initial_usage = None
      initial_trajectory_usages = {}
      if isinstance(raw_init_resp, (str, bytes)):
        init_resp_event = localharness_pb2.OutputEvent()
        json_format.Parse(raw_init_resp, init_resp_event)
        init_resp = init_resp_event.initialize_conversation_response
        initial_history = [
            event_processor.LocalConnectionStep.from_dict(
                json_format.MessageToDict(
                    step_update_proto, preserving_proto_field_name=True
                )
            )
            for step_update_proto in init_resp.history
        ]
        if init_resp.HasField("cumulative_usage"):
          initial_usage = event_processor.parse_usage_metadata(
              init_resp.cumulative_usage
          )
        for entry in init_resp.trajectory_usage:
          if entry.trajectory_id and entry.HasField("usage"):
            initial_trajectory_usages[entry.trajectory_id] = (
                event_processor.parse_usage_metadata(entry.usage)
            )
    except Exception as e:
      process.kill()
      stderr_output = process.stderr.read().decode("utf-8")
      raise RuntimeError(
          f"Failed to initialize conversation at {ws_url}. Stderr:"
          f" {stderr_output}"
      ) from e
    self._connection = LocalConnection(
        process=process,
        ws=ws,
        tool_runner=self._tool_runner,
        hook_runner=self._hook_runner,
        initial_history=initial_history,
        env=self._env,
        debug_config=self._debug_config,
        dynamic_policy_map=self._dynamic_policy_map or None,
        initial_usage=initial_usage,
        initial_trajectory_usages=initial_trajectory_usages,
    )
    self._connection._start_stderr_reader(process.stderr)

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Tears down the backend and releases all resources."""
    if hasattr(self, "_connection") and self._connection:
      await self._connection.disconnect()
      self._connection = None
