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

"""Base interfaces for connections in the Google Antigravity SDK.

A Connection is the SDK's public interface for interacting with an agent
backend, regardless of where the agent runs. Layer 2 APIs (Conversation,
AgentConfig) depend ONLY on this interface — never on transport details.

A ConnectionStrategy knows how to establish a Connection for a specific
backend type and how to tear it down.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

import pydantic

from google.antigravity import types
from google.antigravity.hooks import hooks as hooks_mod
from google.antigravity.hooks import policy
from google.antigravity.triggers import triggers as triggers_mod


class AgentConfig(abc.ABC, pydantic.BaseModel):
  """Abstract base class for agent configuration.

  Each ConnectionStrategy defines a concrete subclass with the
  config fields it needs. Agent introspects the config type to
  auto-dispatch to the correct strategy factory.
  """

  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

  system_instructions: str | types.SystemInstructions | None = None
  capabilities: types.CapabilitiesConfig = pydantic.Field(
      default_factory=lambda: types.CapabilitiesConfig(
          enabled_tools=types.BuiltinTools.read_only()
      )
  )
  tools: list[Callable[..., Any] | str] = pydantic.Field(default_factory=list)
  policies: list[policy.Policy] = pydantic.Field(default_factory=list)
  hooks: list[hooks_mod.Hook] = pydantic.Field(default_factory=list)
  triggers: list[triggers_mod.Trigger] = pydantic.Field(default_factory=list)
  mcp_servers: list[types.McpServerConfig] = pydantic.Field(
      default_factory=list
  )
  workspaces: list[str] = pydantic.Field(default_factory=list)
  conversation_id: str | None = None
  session_continuation_mode: types.SessionContinuationMode | None = None
  save_dir: str | None = None
  app_data_dir: str | None = None
  response_schema: dict[str, Any] | type[pydantic.BaseModel] | str | None = None
  skills_paths: list[str] = pydantic.Field(default_factory=list)
  subagents: list[types.SubagentConfig] = pydantic.Field(default_factory=list)
  debug_config: DebugConfig | None = None
  # Optional retry configuration. Supported by Local, RemoteWebsocket, and JPv2
  # (AntigravityProdActor) connection strategies; ignored by deprecated JPv1.
  retry_config: types.RetryConfig | None = None
  budget_config: types.BudgetConfig | None = None

  @pydantic.field_validator("debug_config", mode="before")
  @classmethod
  def _validate_debug_config(cls, v: Any) -> DebugConfig | None:
    if isinstance(v, bool):
      return DebugConfig() if v else None
    if isinstance(v, dict):
      return DebugConfig(**v)
    return v

  @pydantic.model_validator(mode="after")
  def _apply_debug_settings(self) -> "AgentConfig":
    if self.debug_config is not None and isinstance(
        self.debug_config, DebugConfig
    ):
      self.debug_config.apply_logging()
    return self

  @pydantic.field_validator("conversation_id")
  @classmethod
  def _validate_conversation_id(cls, v: str | None) -> str | None:
    if v:
      if len(v) < 32:
        raise ValueError(
            f"conversation_id must be at least 32 characters long, got {len(v)}"
        )
      if not re.match(r"^[a-zA-Z0-9-]+$", v):
        raise ValueError(
            f"conversation_id must match [a-zA-Z0-9-], got {v!r}"
        )
    return v

  @pydantic.model_validator(mode="after")
  def _validate_session_continuation(self) -> "AgentConfig":
    if (
        self.session_continuation_mode == types.SessionContinuationMode.RESUME
        and not self.conversation_id
    ):
      raise ValueError(
          "conversation_id must be specified when "
          "session_continuation_mode is RESUME"
      )
    return self

  @pydantic.field_validator("response_schema")
  @classmethod
  def _validate_schema(
      cls, v: dict[str, Any] | type[pydantic.BaseModel] | str | None
  ) -> str | None:
    if v is None:
      return None
    if isinstance(v, str):
      try:
        json.loads(v)
        return v
      except json.JSONDecodeError as exc:
        raise ValueError("response_schema string is not valid JSON.") from exc
    if isinstance(v, dict):
      return json.dumps(v)
    if isinstance(v, type) and issubclass(v, pydantic.BaseModel):
      return json.dumps(v.model_json_schema())
    raise ValueError(
        f"Unsupported response_schema format: {type(v).__name__}. "
        "Expected a JSON string, dict, or pydantic.BaseModel subclass."
    )

  @pydantic.field_validator("policies", mode="before")
  @classmethod
  def _validate_policies(cls, v: Any) -> list[policy.Policy]:
    if v is None:
      return []
    if not isinstance(v, (list, tuple, Sequence)) or isinstance(
        v, (str, bytes)
    ):
      v = [v]
    flat_policies = []

    def flatten(item):
      if isinstance(item, (list, tuple, Sequence)) and not isinstance(
          item, (str, bytes)
      ):
        for sub_item in item:
          flatten(sub_item)
      else:
        flat_policies.append(item)

    flatten(v)
    return flat_policies

  def model_copy(
      self, *, update: Mapping[str, Any] | None = None, deep: bool = False
  ) -> "AgentConfig":
    # Override model_copy to prevent deep-copying fields containing callables
    # or stateful objects (tools, hooks, triggers, policies). Deep-copying
    # these fields is destructive because it breaks reference identity (e.g.,
    # duplicating stateful objects that bound methods belong to), making it
    # impossible to observe side-effects on the original instances from the
    # outside.
    copied = super().model_copy(update=update, deep=deep)
    if deep:
      copied.tools = list(self.tools)
      copied.hooks = list(self.hooks)
      copied.triggers = list(self.triggers)
      copied.policies = list(self.policies)
    return copied

  def _get_all_custom_tools(self) -> list[Callable[..., Any]]:
    """Returns all callable custom tools across the main agent and subagents."""
    tools: list[Callable[..., Any]] = []
    seen_names: dict[str, Callable[..., Any]] = {}
    for t in self.tools or []:
      if callable(t):
        name = getattr(t, "__name__", None) or type(t).__name__
        if name in seen_names:
          if seen_names[name] != t:
            raise ValueError(
                f"Duplicate custom tool name '{name}' detected across agent"
                " and subagent configurations."
            )
        else:
          seen_names[name] = t
          tools.append(t)
    for sub in self.subagents or []:
      for tool in sub.tools or []:
        if callable(tool):
          name = getattr(tool, "__name__", None) or type(tool).__name__
          if name in seen_names:
            if seen_names[name] != tool:
              raise ValueError(
                  f"Duplicate custom tool name '{name}' detected across agent"
                  f" and subagent '{sub.name}' configurations."
              )
          else:
            seen_names[name] = tool
            tools.append(tool)
    return tools

  @abc.abstractmethod
  def create_strategy(
      self,
      *,
      # Typed as Any due to circular dep: connection → tool_runner →
      # tool_context → connection.  At runtime these are ToolRunner and
      # HookRunner respectively.
      tool_runner: Any,
      hook_runner: Any,
  ) -> "ConnectionStrategy":
    """Creates the ConnectionStrategy for this config.

    The Agent calls this after setting up ToolRunner, HookRunner,
    and policies. The strategy receives the fully-wired runners.

    Args:
      tool_runner: The fully-wired ToolRunner.
      hook_runner: The fully-wired HookRunner.

    Returns:
      A ConnectionStrategy instance configured with the specified runners.
    """
    ...


class Connection(abc.ABC):
  """A live session with an agent backend.

  This is the common contract that all connection types implement.
  Layer 2 APIs depend only on this interface.
  """

  @property
  def is_idle(self) -> bool:
    """Returns True if the connection is idle and ready for input."""
    return True

  @property
  def _initial_history(self) -> Sequence[types.Step]:
    """Returns the pre-existing session steps restored during handshake."""
    return []

  @property
  def conversation_id(self) -> str:
    """Returns the conversation identifier, or empty string if unset."""
    return ""

  @property
  def debug_config(self) -> DebugConfig | None:
    """Returns the debug configuration for this connection, or None if disabled."""
    return None

  @property
  def cumulative_usage(self) -> types.UsageMetadata:
    """Returns total cumulative token usage from the backend.

    Subclasses override to provide live usage data. Default returns
    an empty UsageMetadata instance (all fields None, indicating no tracking).
    """
    return types.UsageMetadata()

  @property
  def trajectory_usages(self) -> dict[str, types.UsageMetadata]:
    """Returns per-trajectory cumulative token usage from the backend.

    Subclasses override to provide live usage data. Default returns
    an empty dictionary (no tracking).
    """
    return {}

  @property
  def _last_turn_stop_reason(self) -> types.StopReason:
    """Returns the stop reason of the most recent turn.

    Subclasses override to provide live stop reason data. Default returns
    UNSPECIFIED.
    """
    return types.StopReason.UNSPECIFIED

  @abc.abstractmethod
  async def send(self, prompt: types.Content | None, **kwargs: Any) -> None:
    """Sends a prompt to the agent.

    Args:
      prompt: The user message to send.
      **kwargs: Strategy-specific options (model overrides, media, etc.).
    """
    ...

  @abc.abstractmethod
  def receive_steps(self) -> AsyncIterator[types.Step]:
    """Receives steps as they complete from the agent.

    Yields Step objects representing agent actions. The exact fields populated
    depend on the backend, but all steps conform to the Step model.

    Yields:
      Step objects as they occur.
    """
    ...

  async def disconnect(self) -> None:
    """Disconnects the session and releases resources."""
    pass

  async def cancel(self) -> None:
    """Cancels the current turn in progress."""
    pass

  async def wait_for_idle(self) -> None:
    """Blocks until the connection becomes idle."""
    pass

  async def wait_for_wakeup(self, timeout: float = 300.0) -> bool:  # pylint: disable=unused-argument
    """Blocks until the connection wakes up or the timeout is reached.

    Args:
      timeout: Maximum seconds to wait.

    Returns:
      True if the connection woke up, False on timeout.
    """
    return False

  async def _send_tool_results(self, results: list[types.ToolResult]) -> None:
    """Sends tool execution results back to the agent.

    Each connection strategy serializes the results into the backend
    wire format.

    Args:
      results: A list of ToolResult objects.
    """
    pass

  @abc.abstractmethod
  async def send_trigger_notification(self, content: str) -> None:
    """Sends a trigger message to the agent.

    Args:
      content: The trigger message content.
    """
    ...


class ConnectionStrategy(abc.ABC):
  """Strategy for establishing a Connection to an agent backend.

  Each backend type (local, Interactions API, cloud agent) provides its own
  ConnectionStrategy implementation that handles process management,
  transport setup, authentication, and health checking.
  """

  @abc.abstractmethod
  def connect(self) -> Connection:
    """Returns the established Connection.

    Returns:
      The active Connection object.

    Raises:
      RuntimeError: If the connection has not been established.
    """
    # TODO: This method is meant to return a new independent
    # connection, but at the moment most of the implementations return the same
    # connection. This will be rectified in a separate CL.
    ...

  @abc.abstractmethod
  async def __aenter__(self) -> None:
    """Starts the backend and prepares for connections."""
    ...

  @abc.abstractmethod
  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Tears down the backend and releases all resources.

    Args:
      exc_type: The exception type, if any.
      exc_val: The exception value, if any.
      exc_tb: The traceback, if any.
    """
    ...

  @property
  def debug_config(self) -> DebugConfig | None:
    """Returns the debug configuration for this strategy, or None if disabled."""
    return None


class DebugConfig(pydantic.BaseModel):
  """Configuration for client-side and server-side debugging and observability.

  When instantiated with default parameters (`DebugConfig()`), all debug
  features and detailed logging (`logging.DEBUG`) are enabled by default.
  """

  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

  # Whether to enable server-side distributed tracing in the backend.
  enable_server_side_tracing: bool = True
  # Python logging level to apply across SDK modules.
  logging_level: int | str | None = logging.DEBUG

  def apply_logging(self) -> None:
    """Applies the configured logging level to the SDK loggers."""
    if self.logging_level is not None:
      logging.getLogger("google.antigravity").setLevel(self.logging_level)


AgentConfig.model_rebuild()
