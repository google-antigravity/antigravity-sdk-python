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

"""Layer 1 API for Antigravity SDK."""

from collections.abc import Sequence
import contextlib
import logging
from typing import cast

from google.antigravity import types
from google.antigravity.connections import connection as connection_module
from google.antigravity.conversation import conversation as conversation_lib
from google.antigravity.hooks import hook_runner
from google.antigravity.tools import tool_context
from google.antigravity.tools import tool_runner
from google.antigravity.triggers import trigger_runner


__all__ = ["Agent"]


class Agent:
  """High-level Agent API for simplified interaction."""

  def __init__(self, config: connection_module.AgentConfig):
    """Initializes the Agent.

    Args:
        config: Declarative agent configuration.
    """
    self._config = config.model_copy(deep=True)
    if self._config.response_schema:
      # The response_schema is validated/stringified in AgentConfig.
      self._config.capabilities.finish_tool_schema_json = cast(
          str, self._config.response_schema
      )
    self._strategy = None
    self._conversation = None
    self._tool_runner = None
    self._hook_runner = None
    self._trigger_runner = None
    # Use the original config (not self._config) for hooks and triggers:
    # model_copy(deep=True) creates new objects, breaking reference equality
    # for user-provided hooks/triggers. The list() snapshot prevents the
    # caller from mutating our copy, while preserving object identity.
    self._pending_hooks = list(config.hooks)
    self._pending_triggers = list(config.triggers)
    self._exit_stack = contextlib.AsyncExitStack()

  async def __aenter__(self) -> "Agent":
    """Starts the agent session.

    Returns:
        The started Agent instance.
    """
    logging.info("Starting Agent session")
    try:
      self._hook_runner = hook_runner.HookRunner()

      # Register pending hooks
      for hook in self._pending_hooks:
        self._hook_runner.register_hook(hook)
      self._pending_hooks.clear()

      # Apply policies
      active_policies = list(self._config.policies)
      cfg = self._config.capabilities
      read_only_tools = set(types.BuiltinTools.read_only())
      # enabled_tools and disabled_tools are mutually exclusive
      # (enforced by CapabilitiesConfig validation).
      if cfg.enabled_tools is not None:
        active_tools = set(cfg.enabled_tools)
      elif cfg.disabled_tools is not None:
        active_tools = set(types.BuiltinTools) - set(cfg.disabled_tools)
      else:
        active_tools = set(types.BuiltinTools)
      has_write_tools = bool(active_tools - read_only_tools)
      has_mcp_servers = bool(self._config.mcp_servers)
      has_tool_decide_hook = bool(self._hook_runner.pre_tool_call_decide_hooks)

      if (
          (has_write_tools or has_mcp_servers)
          and not active_policies
          and not has_tool_decide_hook
      ):
        raise ValueError(
            "Write tools or MCP servers are enabled without a safety policy. "
            "Add policies=[policy.allow_all()] to approve all tool calls, "
            "or policies=[policy.deny_all(), policy.allow('tool_name')] "
            "to selectively allow specific tools."
        )

      self._tool_runner = tool_runner.ToolRunner(
          tools=self._config._get_all_custom_tools()
      )

      self._strategy = self._config.create_strategy(
          tool_runner=self._tool_runner,
          hook_runner=self._hook_runner,
      )

      logging.info("Starting connection and creating conversation...")
      self._conversation = await self._exit_stack.enter_async_context(
          conversation_lib.Conversation.create(self._strategy)
      )

      # Start triggers via TriggerRunner.
      if self._pending_triggers:
        logging.info("Starting triggers...")
        self._trigger_runner = await self._exit_stack.enter_async_context(
            trigger_runner.TriggerRunner(
                triggers=list(self._pending_triggers),
                connection=self.conversation.connection,
            )
        )
        self._pending_triggers.clear()

      # Wire ToolContext into ToolRunner so tools can access
      # conversation capabilities (same pattern as TriggerRunner).
      if self._tool_runner:
        ctx = tool_context.ToolContext(self.conversation)
        self._tool_runner.set_context(ctx)

      return self
    except Exception:
      logging.exception("Failed to start Agent session, cleaning up...")
      await self._exit_stack.aclose()
      raise

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Stops the agent session.

    Args:
        exc_type: The exception type, if any.
        exc_val: The exception value, if any.
        exc_tb: The traceback, if any.

    Returns:
        True if the exception was suppressed, False or None otherwise.
    """
    logging.info("Stopping Agent session")
    return await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

  async def chat(self, prompt: types.Content) -> types.ChatResponse:
    """Sends a prompt and returns the final response.

    Args:
        prompt: The user prompt or content to send.

    Returns:
        The final response from the agent.

    Raises:
        ValueError: If prompt is None, an empty or whitespace-only string, or an
          empty sequence / sequence containing only empty or whitespace strings.
    """
    if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
      raise ValueError(
          f"chat() requires a non-empty message string. Got: {prompt!r}"
      )
    if (
        isinstance(prompt, Sequence)
        and not isinstance(prompt, str)
        and (
            not prompt
            or all(isinstance(p, str) and not p.strip() for p in prompt)
        )
    ):
      raise ValueError(
          f"chat() requires non-empty message content. Got: {prompt!r}"
      )
    return await self.conversation.chat(prompt)

  @property
  def is_started(self) -> bool:
    """Whether the agent session has been started."""
    return self._conversation is not None

  @property
  def conversation(self) -> conversation_lib.Conversation:
    """Returns the active Conversation session.

    Use this for advanced session introspection: history, turn count,
    compaction indices, usage, or direct send/receive_steps control.
    For most use cases, prefer chat() instead.

    Raises:
      RuntimeError: If the agent session has not been started.
    """
    if not self._conversation:
      raise RuntimeError(
          "Agent session not started. Use 'async with Agent(...)'."
      )
    return self._conversation

  @property
  def conversation_id(self) -> str | None:
    """Returns the conversation identifier assigned by the runtime.

    Available after the session has started and at least one message has
    been exchanged.  Pass this value back via AgentConfig.conversation_id
    to resume from a saved session.  Returns None before the session starts.
    """
    if not self._conversation:
      return None
    return self._conversation.conversation_id or None
