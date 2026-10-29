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

import base64
import contextlib
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

# Server attributes that affect `Engine.create_conversation` initialization
# and must be part of `conversation_cache_key`.
_CONVERSATION_CACHE_SERVER_ATTRS: tuple[str, ...] = (
    "max_output_tokens",
    "thinking_token_budget",
)


class OpenAITool(litert_lm.Tool if _LITERT_AVAILABLE else object):  # type: ignore[misc]
  """Wrapper around LiteRT Tool representing an OpenAPI/Workspace tool."""

  def __init__(self, description: dict[str, Any]):
    self._description = description

  def get_tool_description(self) -> dict[str, Any]:
    return self._description

  def execute(self, param: Any) -> Any:
    raise NotImplementedError("Proxy tools are not executable.")

  def __eq__(self, other: Any) -> bool:
    """Compares tool schemas by value for conversation cache key matching.

    New `OpenAITool` instances are created on every HTTP request when parsing
    the JSON `tools` payload. Value-based equality ensures `tools` lists
    compare equal across consecutive turns in `conversation_cache_key`.

    Args:
      other: Object to compare against.

    Returns:
      True if `other` is an `OpenAITool` with an identical tool description.
    """
    if not isinstance(other, OpenAITool):
      return False
    return self._description == other._description


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
  """Thread-safe HTTP Server holding references to LiteRT Engine context.

  Attributes:
    engine: The LiteRT-LM Engine instance.
    model_name: The model identifier string.
    max_output_tokens: Maximum tokens to generate per turn.
    thinking_token_budget: Thinking token budget or None.
    engine_lock: Mutex protecting non-thread-safe engine calls.
    litert_lm_conversation: The active LiteRT-LM `Conversation` cached across
      turns so its internal KV cache can be reused, or `None`.
    conversation_cache_key: Tuple `(expected_messages, tools, *server_attrs)`
      representing the full message history (including the last prompt and
      assistant response) and config currently held in `litert_lm_conversation`,
      or `None`.
  """

  def __init__(
      self,
      server_address: tuple[str, int],
      RequestHandlerClass: type[http.server.BaseHTTPRequestHandler],
      engine: Any,
      model_name: str,
      max_output_tokens: int = 16384,
      thinking_token_budget: int | None = 8192,
  ):
    self.engine = engine
    self.model_name = model_name
    self.max_output_tokens = max_output_tokens
    self.thinking_token_budget = thinking_token_budget
    self.engine_lock = threading.Lock()
    self.litert_lm_conversation: Any = None
    self.conversation_cache_key: tuple[Any, ...] | None = None
    super().__init__(server_address, RequestHandlerClass)

  def close_conversation(self) -> None:
    """Closes and resets the currently cached conversation, if any."""
    if self.litert_lm_conversation is not None:
      try:
        self.litert_lm_conversation.__exit__(None, None, None)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error closing LiteRT conversation: %s", e)
      finally:
        self.litert_lm_conversation = None
    self.conversation_cache_key = None

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


def _translate_openai_content(
    content: str | list[str | dict[str, Any]] | None,
) -> Any:
  """Translates OpenAI message content into LiteRT-LM format.

  See OpenAI Chat Completions message format reference:
  https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

  When `content` is a list, each element can be a plain `str` or a `dict` with
  one of the following structures:

  - Text: `{"type": "text", "text": "..."}`
  - Image URL / data URI:
    `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}`
    or `{"type": "image_url", "image_url": {"url": "file:///path/to/img"}}`
  - Image part: `{"type": "image", "blob": "<base64>"}` or
    `{"type": "image", "path": "/path/to/img"}`
  - Input audio: `{"type": "input_audio", "input_audio": {"data": "<base64>"}}`
  - Audio part: `{"type": "audio", "blob": "<base64>"}` or
    `{"type": "audio", "path": "/path/to/audio"}`

  Args:
    content: Plain text string, `None`, or a list of multimodal content parts.

  Returns:
    A plain `str` when `content` is a string or `None` (or when `litert_lm` is
    unavailable), or a `litert_lm.Contents` instance when `content` is a list
    of content parts.

  Raises:
    ValueError: If a content part in `content` is malformed or has an
      unsupported type.
  """
  if litert_lm is None or not isinstance(content, list):
    return content or ""

  parts = []
  for part in content:
    if isinstance(part, str):
      parts.append(litert_lm.Content.Text(part))
    elif isinstance(part, dict):
      part_type = part.get("type")
      if part_type == "text":
        parts.append(litert_lm.Content.Text(part.get("text", "")))
      elif part_type == "image_url":
        image_url = part.get("image_url", {})
        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
        if url.startswith("data:"):
          if "," not in url:
            raise ValueError("Invalid data URL format")
          header, data = url.split(",", 1)
          if "base64" not in header:
            raise ValueError(
                "Unsupported data URL format (only base64 is supported)"
            )
          parts.append(litert_lm.Content.ImageBytes(base64.b64decode(data)))
        else:
          path = url[7:] if url.startswith("file://") else url
          if not path:
            raise ValueError("Empty image_url")
          parts.append(litert_lm.Content.ImageFile(path))
      elif part_type == "image":
        if "blob" in part:
          parts.append(
              litert_lm.Content.ImageBytes(base64.b64decode(part["blob"]))
          )
        elif "path" in part:
          parts.append(litert_lm.Content.ImageFile(part["path"]))
        else:
          raise ValueError(
              "Invalid image content part: missing 'blob' or 'path'"
          )
      elif part_type == "input_audio":
        input_audio = part.get("input_audio", {})
        data = (
            input_audio.get("data", "") if isinstance(input_audio, dict) else ""
        )
        if not data:
          raise ValueError("Empty input_audio data")
        parts.append(litert_lm.Content.AudioBytes(base64.b64decode(data)))
      elif part_type == "audio":
        if "blob" in part:
          parts.append(
              litert_lm.Content.AudioBytes(base64.b64decode(part["blob"]))
          )
        elif "path" in part:
          parts.append(litert_lm.Content.AudioFile(part["path"]))
        else:
          raise ValueError(
              "Invalid audio content part: missing 'blob' or 'path'"
          )
      else:
        raise ValueError(f"Unsupported content part type: {part_type!r}")
    else:
      raise ValueError(f"Unsupported content part: {type(part)}")

  return litert_lm.Contents.of(parts)


def _build_assistant_message(
    content: str | list[str | dict[str, Any]] | None,
    tool_calls_payload: Any = None,
) -> Any:
  """Builds a normalized LiteRT-LM assistant Message from text and tool calls.

  Used both when translating prior assistant messages from an incoming OpenAI
  request and when recording the newly generated assistant turn into
  `conversation_cache_key`, ensuring both produce identical `Message` objects.

  Args:
    content: Assistant text string, `None`, or a list of OpenAI multimodal
      content parts (see `_translate_openai_content` for supported part
      schemas).
    tool_calls_payload: Optional list of OpenAI-formatted tool call dicts.

  Returns:
    A `litert_lm.Message` representing the assistant turn, or `None` if
    `litert_lm` is unavailable.
  """
  if litert_lm is None:
    return None
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
  if isinstance(content, list):
    content_obj = _translate_openai_content(content)
  elif content:
    content_obj = litert_lm.Contents.of(content)
  else:
    content_obj = litert_lm.Contents.empty()
  return litert_lm.Message.model(
      contents=content_obj,
      tool_calls=litert_tool_calls,
  )


def _extract_litert_fields(payload: Any) -> tuple[str, str, list[Any]]:
  """Extracts text, thought reasoning, and tool calls from a LiteRT message dict."""
  text_output = ""
  reasoning_output = ""
  tool_calls: list[Any] = []
  if isinstance(payload, dict):
    channels = payload.get("channels", {})
    if isinstance(channels, dict):
      thought = channels.get("thought", "")
      if isinstance(thought, str):
        reasoning_output = thought
    text_output = "".join(
        item.get("text", "")
        for item in payload.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    )
    tool_calls = payload.get("tool_calls", [])
  return text_output, reasoning_output, tool_calls


def _make_conversation_cache_key(
    messages: list[Any],
    tools: list[Any] | None,
    server: LiteRTOpenAIServer,
) -> tuple[Any, ...]:
  """Constructs the cache key tuple for conversation reuse."""
  return (
      messages,
      tools,
      *(getattr(server, attr) for attr in _CONVERSATION_CACHE_SERVER_ATTRS),
  )


@contextlib.contextmanager
def _get_or_create_conversation(
    server: LiteRTOpenAIServer,
    context_messages: list[Any],
    prompt: Any,
    tools: list[Any] | None,
    thinking_config: Any,
) -> Any:
  """Yields a cached or newly created LiteRT Conversation and a save callback.

  OpenAI `/v1/chat/completions` requests are stateless and resend the full
  message history on every turn, whereas LiteRT-LM's `Conversation` is stateful.
  Each request splits the translated messages into `context_messages`
  (`litert_messages[:-1]`, all prior turns) and `prompt` (`litert_messages[-1]`,
  the new user or tool turn).

  If `(context_messages, tools, *server_attrs)` matches
  `server.conversation_cache_key` saved at the end of the previous turn
  (`[*prev_context_messages, prev_prompt, prev_assistant_msg]`), the existing
  `Conversation` and its KV cache are reused and only `prompt` is sent.
  Otherwise, the old conversation is closed and a new one is created with
  `context_messages`.
  """
  cache_key = _make_conversation_cache_key(context_messages, tools, server)
  if (
      server.litert_lm_conversation is not None
      and server.conversation_cache_key == cache_key
  ):
    logging.info(
        "LiteRT conversation cache hit: reusing conversation with %d context"
        " messages",
        len(context_messages),
    )
    conv = server.litert_lm_conversation
  else:
    logging.info(
        "LiteRT conversation cache miss: creating new conversation with %d"
        " context messages",
        len(context_messages),
    )
    server.close_conversation()
    constrained_decoding_config = (
        litert_lm.ConstrainedDecodingConfig() if litert_lm is not None else None
    )
    conv = server.engine.create_conversation(
        messages=context_messages,
        tools=tools,
        automatic_tool_calling=False,
        max_output_tokens=server.max_output_tokens,
        thinking_config=thinking_config,
        constrained_decoding_config=constrained_decoding_config,
    ).__enter__()
    server.litert_lm_conversation = conv

  saved_assistant: list[Any] = [None]

  def save_assistant(assistant_msg: Any) -> None:
    saved_assistant[0] = assistant_msg

  try:
    yield conv, save_assistant
  except Exception:
    if hasattr(conv, "cancel_process"):
      try:
        conv.cancel_process()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    server.close_conversation()
    raise
  else:
    if saved_assistant[0] is not None:
      server.conversation_cache_key = _make_conversation_cache_key(
          [*context_messages, prompt, saved_assistant[0]],
          tools,
          server,
      )
    else:
      server.close_conversation()


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
      pending_tool_responses: list[Any] = []
      for idx, m in enumerate(messages):
        if not isinstance(m, dict):
          self.send_error(400, "Corrupted history frame")
          return
        role = m.get("role")
        raw_content = m.get("content")
        if role in ("system", "developer"):
          litert_messages.append(
              litert_lm.Message.system(_translate_openai_content(raw_content))
          )
        elif role == "user":
          litert_messages.append(
              litert_lm.Message.user(_translate_openai_content(raw_content))
          )
        elif role in ("assistant", "model"):
          tool_calls_payload = m.get("tool_calls", [])
          litert_messages.append(
              _build_assistant_message(raw_content, tool_calls_payload)
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
          # Merge consecutive `role == "tool"` messages into a single
          # `Message.tool` turn so parallel tool calls in a single assistant
          # turn match the cached LiteRT conversation history.
          tool_content = raw_content or ""
          pending_tool_responses.append(
              litert_lm.Content.ToolResponse(name=name, response=tool_content)
          )
          next_msg = messages[idx + 1] if idx + 1 < len(messages) else None
          if not isinstance(next_msg, dict) or next_msg.get("role") != "tool":
            litert_messages.append(
                litert_lm.Message.tool(
                    litert_lm.Contents.of(pending_tool_responses)
                )
            )
            pending_tool_responses = []
    except Exception as e:  # pylint: disable=broad-exception-caught
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

    thinking_config = None
    if server.thinking_token_budget is not None:
      thinking_config = litert_lm.ThinkingConfig(
          thinking_token_budget=server.thinking_token_budget
      )

    # Thread-safe serialization for non-thread-safe C++ engine model inferences
    with server.engine_lock:
      try:
        with _get_or_create_conversation(
            server=server,
            context_messages=context_messages,
            prompt=prompt,
            tools=litert_tools or None,
            thinking_config=thinking_config,
        ) as (conv, save_assistant):
          if stream:
            assistant_msg = self._stream_response(
                conv, prompt, server.model_name
            )
          else:
            assistant_msg = self._handle_synchronous(
                conv, prompt, server.model_name
            )
          save_assistant(assistant_msg)
      except Exception as e:  # pylint: disable=broad-exception-caught
        self.close_connection = True
        if _is_connection_reset_error(e) or isinstance(e, ConnectionError):
          logging.warning(
              "Client disconnected early during completions stream."
          )
        else:
          logging.exception("Error occurred during LiteRT inference execution")
          try:
            self.send_error(500, f"Inference Engine error: {e}")
          except Exception:  # pylint: disable=broad-exception-caught
            pass

  def _stream_response(self, conv: Any, prompt: Any, model_name: str) -> Any:
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
    text_parts: list[str] = []
    all_tool_calls: list[dict[str, Any]] = []
    # Yield content delta stream
    for chunk in conv.send_message_async(prompt):
      text_output, reasoning_output, tool_calls = _extract_litert_fields(chunk)

      if text_output:
        text_parts.append(text_output)

      if text_output or reasoning_output or tool_calls:
        delta: dict[str, Any] = {}
        if reasoning_output:
          delta["reasoning"] = reasoning_output
          delta["reasoning_content"] = reasoning_output
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
          all_tool_calls.extend(delta["tool_calls"])
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
    return _build_assistant_message("".join(text_parts), all_tool_calls)

  def _handle_synchronous(self, conv: Any, prompt: Any, model_name: str) -> Any:
    """Handles synchronous completions and returns raw JSON payload response."""
    response = conv.send_message(prompt)
    text_output, reasoning_output, tool_calls = _extract_litert_fields(response)

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
    if reasoning_output:
      message_payload["reasoning"] = reasoning_output
      message_payload["reasoning_content"] = reasoning_output
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
    return _build_assistant_message(text_output, openai_tool_calls)
