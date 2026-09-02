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

"""LiteRT OpenAI HTTP Server loopback."""

import http.server
import json
import logging
import sys
import threading
import time
from typing import Any, cast

try:
  # pylint: disable=g-import-not-at-top
  import litert_lm  # type: ignore[import-error]

  _LITERT_AVAILABLE = True
except ImportError:
  litert_lm = None  # type: ignore[assignment]
  _LITERT_AVAILABLE = False


class OpenAITool(litert_lm.Tool if _LITERT_AVAILABLE else object):  # type: ignore[misc]
  """Wrapper around LiteRT Tool representing an OpenAPI/Workspace tool."""

  def __init__(self, description: dict[str, Any]):
    self._description = description

  def get_tool_description(self) -> dict[str, Any]:
    return self._description

  def execute(self, param: Any) -> Any:
    raise NotImplementedError("Proxy tools are not executable.")


def _is_connection_reset_error(exc: BaseException | None) -> bool:
  """Returns True if the exception indicates a client socket disconnect."""
  if exc is None:
    return False
  if isinstance(
      exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)
  ):
    return True
  if isinstance(exc, OSError) and getattr(exc, "winerror", None) in (
      10054,
      10053,
  ):
    return True
  return False


class LiteRTOpenAIServer(http.server.ThreadingHTTPServer):
  """Thread-safe HTTP Server holding references to LiteRT Engine context."""

  def __init__(
      self,
      server_address: tuple[str, int],
      RequestHandlerClass: type[http.server.BaseHTTPRequestHandler],
      engine: Any,
      model_name: str,
  ):
    self.engine = engine
    self.model_name = model_name
    self.engine_lock = threading.Lock()
    super().__init__(server_address, RequestHandlerClass)

  def handle_error(
      self, request: Any, client_address: tuple[str, int] | str
  ) -> None:
    """Suppress connection reset tracebacks when client disconnects early."""
    _, exc_val, _ = sys.exc_info()
    if _is_connection_reset_error(exc_val):
      logging.debug("Client %s disconnected (%s)", client_address, exc_val)
      return
    super().handle_error(request, client_address)


def _format_openai_tool_call(
    tc: Any, index: int, timestamp_str: str
) -> dict[str, Any]:
  """Formats a model-generated tool call object into OpenAI format safely."""
  if isinstance(tc, dict):
    func_info = tc.get("function", {})
    if not isinstance(func_info, dict):
      func_info = {}
    name = func_info.get("name", "")
    args = func_info.get("arguments", {})
  elif hasattr(tc, "name"):
    name = getattr(tc, "name", "")
    args = getattr(tc, "arguments", {})
  else:
    name = ""
    args = {}

  if isinstance(args, str):
    arguments_str = args
  else:
    arguments_str = json.dumps(args if isinstance(args, dict) else {})

  return {
      "id": f"call_{timestamp_str}_{index}",
      "type": "function",
      "function": {
          "name": name,
          "arguments": arguments_str,
      },
  }


class LiteRTOpenAIHandler(http.server.BaseHTTPRequestHandler):
  """Lightweight HTTP Request Handler translating OpenAI API to litert_lm."""

  def log_message(self, format_str: str, *args: Any) -> None:
    logging.debug(format_str, *args)

  def address_string(self) -> str:
    """Bypass reverse DNS hostname resolution to prevent hangs in network-isolated sandboxes."""
    return self.client_address[0]

  def handle(self) -> None:
    """Handle HTTP requests, catching connection resets gracefully."""
    try:
      super().handle()
    except (
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
        OSError,
    ) as e:
      if _is_connection_reset_error(e):
        self.close_connection = True
      else:
        raise

  # pylint: disable=invalid-name
  def do_GET(self) -> None:
    """Handles HTTP GET models requests."""
    server = cast(LiteRTOpenAIServer, self.server)
    if self.path in ("/v1/models", "/v1/models/"):
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      response_data = {
          "object": "list",
          "data": [{
              "id": server.model_name,
              "object": "model",
              "created": int(time.time()),
              "owned_by": "litert-lm",
          }],
      }
      self.wfile.write(json.dumps(response_data).encode("utf-8"))
    else:
      self.send_error(404, "Not Found")

  # pylint: disable=invalid-name
  def do_POST(self) -> None:
    """Handles HTTP POST completions requests."""
    if litert_lm is None:
      self.send_error(500, "LiteRT-LM is not available.")
      return
    server = cast(LiteRTOpenAIServer, self.server)
    if self.path not in ("/v1/chat/completions", "/v1/chat/completions/"):
      self.send_error(404, "Not Found")
      return

    try:
      content_length = int(self.headers.get("Content-Length", 0))
      body_bytes = self.rfile.read(content_length)
      body = json.loads(body_bytes.decode("utf-8"))
    # pylint: disable=broad-exception-caught
    except Exception as e:  # pylint: disable=broad-exception-caught  # pylint: disable=broad-exception-caught
      self.send_error(400, f"Invalid JSON payload: {e}")
      return

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not messages:
      self.send_error(400, "Missing messages")
      return

    try:
      litert_messages = []
      for idx, m in enumerate(messages):
        if not isinstance(m, dict):
          self.send_error(400, "Corrupted history frame")
          return
        role = m.get("role")
        content = m.get("content") or ""
        if role in ("system", "developer"):
          litert_messages.append(litert_lm.Message.system(content))
        elif role == "user":
          litert_messages.append(litert_lm.Message.user(content))
        elif role in ("assistant", "model"):
          tool_calls_payload = m.get("tool_calls", [])
          if not isinstance(tool_calls_payload, list):
            tool_calls_payload = []
          litert_tool_calls = []
          for tc in tool_calls_payload:
            if isinstance(tc, dict):
              tc_func = tc.get("function", {})
              if isinstance(tc_func, dict):
                args_raw = tc_func.get("arguments", "{}")
                if isinstance(args_raw, dict):
                  tc_args = args_raw
                else:
                  try:
                    tc_args = json.loads(args_raw)
                  except (ValueError, TypeError):
                    tc_args = {}
                litert_tool_calls.append(
                    litert_lm.ToolCall(
                        name=tc_func.get("name", ""),
                        arguments=tc_args,
                    )
                )
          content_obj = (
              litert_lm.Contents.of(content)
              if content
              else litert_lm.Contents.empty()
          )
          litert_messages.append(
              litert_lm.Message.model(
                  contents=content_obj,
                  tool_calls=litert_tool_calls,
              )
          )
        elif role == "tool":
          name = m.get("name")
          if not name:
            tool_call_id = m.get("tool_call_id")
            if tool_call_id:
              for prev_m in reversed(messages[:idx]):
                if prev_m.get("role") in ("assistant", "model"):
                  for tc in prev_m.get("tool_calls", []):
                    if isinstance(tc, dict) and tc.get("id") == tool_call_id:
                      tc_func = tc.get("function", {})
                      if isinstance(tc_func, dict):
                        name = tc_func.get("name")
                        break
                if name:
                  break
          if not name:
            name = "dummy_tool"
          # Correct tool responses require ToolResponse content wrapper
          tool_resp = litert_lm.Content.ToolResponse(
              name=name, response=content
          )
          litert_messages.append(
              litert_lm.Message.tool(litert_lm.Contents.of(tool_resp))
          )
    # pylint: disable=broad-exception-caught
    except Exception as e:  # pylint: disable=broad-exception-caught  # pylint: disable=broad-exception-caught
      self.send_error(400, f"Failed to translate messages: {e}")
      return

    context_messages = litert_messages[:-1] if litert_messages else []
    prompt = litert_messages[-1] if litert_messages else ""

    tools_payload = body.get("tools", [])
    litert_tools = []
    if tools_payload and isinstance(tools_payload, list):

      for t in tools_payload:
        if (
            isinstance(t, dict)
            and t.get("type") == "function"
            and isinstance(t.get("function"), dict)
            and isinstance(t["function"].get("name"), str)
        ):
          litert_tools.append(OpenAITool(t))

    # Thread-safe serialization for non-thread-safe C++ engine model inferences
    with server.engine_lock:
      try:
        with server.engine.create_conversation(
            messages=context_messages,
            tools=litert_tools or None,
            automatic_tool_calling=False,
            max_output_tokens=16384,
            thinking_config=litert_lm.ThinkingConfig(
                thinking_token_budget=8192
            ),
            constrained_decoding_config=litert_lm.ConstrainedDecodingConfig(),
        ) as conv:
          if stream:
            self._stream_response(conv, prompt, server.model_name)
          else:
            self._handle_synchronous(conv, prompt, server.model_name)
      except ConnectionError:
        logging.warning("Client disconnected early during completions stream.")
      # pylint: disable=broad-exception-caught
      except Exception as e:  # pylint: disable=broad-exception-caught  # pylint: disable=broad-exception-caught
        logging.exception("Error occurred during LiteRT inference execution")
        try:
          self.send_error(500, f"Inference Engine error: {e}")
        # pylint: disable=broad-exception-caught
        except Exception:
          pass

  def _stream_response(self, conv: Any, prompt: Any, model_name: str) -> None:
    """Streams completions delta chunks to client using Server-Sent Events (SSE)."""
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.send_header("Connection", "keep-alive")
    self.end_headers()

    chunk_id = f"chatcmpl_{int(time.time())}"
    created_ts = int(time.time())

    # Send initial assistant role delta
    initial_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }
    self.wfile.write(f"data: {json.dumps(initial_chunk)}\n\n".encode("utf-8"))
    self.wfile.flush()

    finish_reason = "stop"
    # Yield content delta stream
    for chunk in conv.send_message_async(prompt):
      text_output = ""
      tool_calls = []
      if isinstance(chunk, dict):
        text_output = "".join(
            item.get("text", "")
            for item in chunk.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        )
        tool_calls = chunk.get("tool_calls", [])

      if text_output or tool_calls:
        delta = {}
        if text_output:
          delta["content"] = text_output
        if tool_calls:
          now_str = str(int(time.time()))
          delta["tool_calls"] = [
              {
                  **_format_openai_tool_call(tc, i, now_str),
                  "index": i,
              }
              for i, tc in enumerate(tool_calls)
          ]
          finish_reason = "tool_calls"

        delta_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }],
        }
        self.wfile.write(f"data: {json.dumps(delta_chunk)}\n\n".encode("utf-8"))
        self.wfile.flush()

    # Write terminal frames
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    }
    self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
    self.wfile.write(b"data: [DONE]\n\n")
    self.wfile.flush()

  def _handle_synchronous(
      self, conv: Any, prompt: Any, model_name: str
  ) -> None:
    """Handles synchronous completions and returns raw JSON payload response."""
    response = conv.send_message(prompt)
    text_output = ""
    tool_calls = []
    if isinstance(response, dict):
      text_output = "".join(
          item.get("text", "")
          for item in response.get("content", [])
          if isinstance(item, dict) and item.get("type") == "text"
      )
      tool_calls = response.get("tool_calls", [])

    now_str = str(int(time.time()))
    openai_tool_calls = [
        _format_openai_tool_call(tc, i, now_str)
        for i, tc in enumerate(tool_calls)
    ]

    finish_reason = "tool_calls" if openai_tool_calls else "stop"

    message_payload: dict[str, Any] = {
        "role": "assistant",
        "content": text_output or None,
    }
    if openai_tool_calls:
      message_payload["tool_calls"] = openai_tool_calls
      if not text_output:
        message_payload.pop("content", None)

    resp_body = {
        "id": f"chatcmpl_{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": message_payload,
            "finish_reason": finish_reason,
        }],
    }
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(json.dumps(resp_body).encode("utf-8"))
