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

"""Tool call policy system for the Google Antigravity SDK.

Provides a declarative API for expressing tool call policies (APPROVE, DENY,
ASK_USER) that are enforced via the hooks system. Policies are evaluated using
a priority-based model where specificity and safety determine precedence:

  Specific Deny > Specific Ask > Specific Allow >
  Wildcard Deny > Wildcard Ask > Wildcard Allow

Within each priority group, first match wins, enabling short-circuit evaluation.

Default Behavior:

  ``LocalAgentConfig`` uses ``confirm_run_command()`` as its default policy.
  This denies ``run_command`` (the most dangerous tool) while allowing all
  other tools.  To enable autonomous shell access, explicitly pass
  ``policies=[policy.allow_all()]``.

Policy Denial vs. Disabling Tools:

  Policies operate at the hook layer: a denied tool is still *visible* to the
  model in its tool list. If the model calls a policy-denied tool, the SDK
  rejects the call and returns a denial message. The model may then retry or
  choose another approach, but each attempt costs tokens.

  To remove a tool from the model's context entirely — so it never sees the
  tool and never wastes tokens on it — use ``CapabilitiesConfig.disabled_tools``
  (or ``enabled_tools``) instead.

  **Use policies** when the restriction is conditional or context-dependent
  (e.g., denying ``run_command`` only for dangerous arguments, or requiring
  user approval for certain operations).

  **Use CapabilitiesConfig** when the tool is simply irrelevant to the agent's
  purpose and should not appear in its context at all.

Usage:
  from google.antigravity.hooks import policy

  policies = [
      policy.deny("*"),                     # Block everything by default
      policy.allow("read_file"),            # Except reading files
      policy.deny("run_command",            # Block dangerous commands
          when=lambda args: "rm" in args.get("CommandLine", "")),
      policy.ask_user("run_command",        # Ask for other commands
          handler=my_approval_fn),
  ]

  hook = policy.enforce(policies)
  # Register hook with HookRunner's pre_tool_call_decide_hooks
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
import dataclasses
import enum
import inspect
import logging
import os
import typing
from typing import Any, Union, overload

import pydantic

from google.antigravity.proto import localharness_pb2
from google.antigravity import types
from google.antigravity.hooks import hooks


_logger = logging.getLogger(__name__)

# A predicate receives the tool call's argument dict (or a Pydantic model,
# if the predicate's first parameter is annotated with a BaseModel subclass)
# and returns whether the policy applies. Supports both sync and async.
Predicate = Callable[..., bool | Awaitable[bool]]

# An ask_user handler receives the full ToolCall and returns whether the
# user approved execution. Supports both sync and async callables.
AskUserHandler = Callable[[types.ToolCall], bool | Awaitable[bool]]

_WILDCARD = "*"
WORKSPACE_ONLY_POLICY_NAME = "workspace_only"
_WORKSPACE_ONLY_POLICY_NAME = WORKSPACE_ONLY_POLICY_NAME


class Decision(enum.Enum):
  """Outcome a policy can produce."""

  APPROVE = "APPROVE"
  DENY = "DENY"
  ASK_USER = "ASK_USER"


@dataclasses.dataclass(frozen=True)
class Policy:
  """A single tool call policy rule.

  Attributes:
    tool: Tool name this policy targets, or "*" for all tools.
    decision: The outcome when this policy matches.
    when: Optional predicate on the tool call's arguments. If None the policy
      matches any call to the named tool.
    ask_user: Handler invoked when decision is ASK_USER. Must be provided for
      ASK_USER policies (validated at enforce() time).
    name: Human-readable label used in logging and deny reasons.
  """

  tool: str
  decision: Decision
  when: Predicate | None = None
  ask_user: AskUserHandler | None = None
  name: str = ""


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _mcp_policies(
    decision: Decision,
    mcp_config: types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
    handler: AskUserHandler | None = None,
) -> list[Policy]:
  """Generates MCP-specific policies.

  Commonly used by allow(), deny(), and ask_user() builders. Translates the
  config and tools into the structured 'server/tool' or 'server/*' target
  formats expected by the runtime hook.

  Args:
    decision: The Decision outcome when the policy matches.
    mcp_config: The BaseMcpServerConfig of the target MCP server.
    mcp_tools: Optional sequence of tool names to allow/deny/ask for. If None,
      applies to all tools on this server.
    when: Optional argument predicate.
    name: Optional human-readable label.
    handler: Optional AskUserHandler for ASK_USER policies.

  Returns:
    A list of Policy objects.
  """
  server = mcp_config.name

  if isinstance(mcp_tools, str):
    raise ValueError(
        f"mcp_tools must be a sequence of strings (e.g., ['{mcp_tools}']), "
        "not a single string."
    )

  if mcp_tools is None:
    # Server-wide wildcard policy (covers all tools of this MCP server).
    # Lowercase prefix matching is purely for debug-logging consistency.
    policy_name = name or f"{decision.value.lower()}_{server}_all"
    return [
        Policy(
            tool=f"{server}/*",
            decision=decision,
            when=when,
            name=policy_name,
            ask_user=handler,
        )
    ]

  policies = []
  for t in mcp_tools:
    policy_name = (
        f"{name}_{t}" if name else f"{decision.value.lower()}_{server}_{t}"
    )
    policies.append(
        Policy(
            tool=f"{server}/{t}",
            decision=decision,
            when=when,
            name=policy_name,
            ask_user=handler,
        )
    )
  return policies


@overload
def allow(
    tool: str,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Policy:
  ...


@overload
def allow(
    tool: types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> list[Policy]:
  ...


def allow(
    tool: str | types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Any:
  """Creates an APPROVE policy.

  Args:
    tool: Tool name, "*" for all tools, or BaseMcpServerConfig.
    mcp_tools: Optional list of tool names if BaseMcpServerConfig is provided.
    when: Optional argument predicate.
    name: Human-readable label.

  Returns:
    A Policy or a list of Policies.
  """
  if isinstance(tool, str):
    if mcp_tools is not None:
      raise ValueError("mcp_tools cannot be specified when tool is a string.")
    return Policy(tool=tool, decision=Decision.APPROVE, when=when, name=name)  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.

  return _mcp_policies(Decision.APPROVE, tool, mcp_tools, when=when, name=name)  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.


@overload
def deny(
    tool: str,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Policy:
  ...


@overload
def deny(
    tool: types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> list[Policy]:
  ...


def deny(
    tool: str | types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Any:
  """Creates a DENY policy.

  Args:
    tool: Tool name, "*" for all tools, or BaseMcpServerConfig.
    mcp_tools: Optional list of tool names if BaseMcpServerConfig is provided.
    when: Optional argument predicate.
    name: Human-readable label.

  Returns:
    A Policy or a list of Policies.
  """
  if isinstance(tool, str):
    if mcp_tools is not None:
      raise ValueError("mcp_tools cannot be specified when tool is a string.")
    return Policy(tool=tool, decision=Decision.DENY, when=when, name=name)  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.

  return _mcp_policies(Decision.DENY, tool, mcp_tools, when=when, name=name)  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.


@overload
def ask_user(
    tool: str,
    *,
    handler: AskUserHandler | None = None,
    when: Predicate | None = None,
    name: str = "",
) -> Policy:
  ...


@overload
def ask_user(
    tool: types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    handler: AskUserHandler | None = None,
    when: Predicate | None = None,
    name: str = "",
) -> list[Policy]:
  ...


def ask_user(
    tool: str | types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    handler: AskUserHandler | None = None,
    when: Predicate | None = None,
    name: str = "",
) -> Any:
  """Creates an ASK_USER policy.

  Args:
    tool: Tool name, "*" for all tools, or BaseMcpServerConfig.
    mcp_tools: Optional list of tool names if BaseMcpServerConfig is provided.
    handler: Optional callable invoked to obtain user approval for the tool
      call. If omitted, confirmation is delegated to the host platform.
    when: Optional argument predicate.
    name: Human-readable label.

  Returns:
    A Policy or a list of Policies.
  """
  if isinstance(tool, str):
    if mcp_tools is not None:
      raise ValueError("mcp_tools cannot be specified when tool is a string.")
    return Policy(  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.
        tool=tool,
        decision=Decision.ASK_USER,
        when=when,
        ask_user=handler,
        name=name,
    )

  return _mcp_policies(  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.
      Decision.ASK_USER,
      tool,
      mcp_tools,
      when=when,
      name=name,
      handler=handler,
  )


def allow_all() -> Policy:
  """Creates a policy that approves all tool calls without confirmation.

  Intended for autonomous agents and local development where interactive
  confirmation is not needed. Equivalent to ``allow("*")``.

  Returns:
    A Policy that approves every tool call.
  """
  return allow(_WILDCARD, name="allow_all")


def safe_defaults(handler: AskUserHandler) -> list[Policy]:
  """Creates a set of safe default policies.

  Allows all read-only tools and asks the user for any other tool calls.

  Args:
    handler: The handler to invoke for ASK_USER decisions.

  Returns:
    A list of Policies.
  """
  return [allow(t.value) for t in types.BuiltinTools.read_only()] + [
      ask_user("*", handler=handler)
  ]


def deny_all() -> Policy:
  """Creates a policy that denies all tool calls.

  Use as a base rule with specific ``allow()`` overrides for a
  deny-by-default posture. Specific policies always take priority over
  wildcard policies, so ``[deny_all(), allow("view_file")]`` will allow
  only ``view_file`` and deny everything else.

  Returns:
    A Policy that denies every tool call.
  """
  return deny(_WILDCARD, name="deny_all")


def confirm_run_command(
    handler: AskUserHandler | None = None,
) -> list[Policy]:
  """Safe default: allows all tools, denies or confirms run_command.

  When no handler is given, ``run_command`` is denied outright — the agent
  sees the tool but calls are rejected with a clear message explaining
  how to enable it.  When a handler is given, ``run_command`` calls
  trigger an ASK_USER flow instead.

  All other tools (file read/write, subagents, image generation, etc.)
  are allowed.

  This is the default policy for ``LocalAgentConfig``.

  Args:
    handler: Optional handler for ASK_USER on run_command. If None, run_command
      is denied.

  Returns:
    A list of Policies.
  """
  if handler is not None:
    return [
        ask_user(
            types.BuiltinTools.RUN_COMMAND.value,
            handler=handler,
            name="confirm_run_command",
        ),
        allow(_WILDCARD, name="confirm_run_command"),
    ]
  return [
      deny(types.BuiltinTools.RUN_COMMAND.value, name="confirm_run_command"),
      allow(_WILDCARD, name="confirm_run_command"),
  ]


PathOrStr = Union[str, os.PathLike[str]]


def workspace_only(workspaces: Sequence[PathOrStr]) -> list[Policy]:
  """Restricts file tools to the given workspace directories.

  File read/write/create operations targeting paths outside any of the
  configured workspace directories are denied. Other tools are unaffected.

  Args:
    workspaces: Absolute paths of allowed workspace directories.

  Returns:
    A list of Policies.
  """
  del workspaces
  file_tools = [t.value for t in types.BuiltinTools.file_tools()]
  return [deny(tool, name=_WORKSPACE_ONLY_POLICY_NAME) for tool in file_tools]


# ---------------------------------------------------------------------------
# Dynamic evaluation helpers
# ---------------------------------------------------------------------------


async def _evaluate_predicate(
    policy: Policy, tool_call: types.ToolCall
) -> bool:
  """Evaluates a policy's predicate.

  If the predicate is None, the policy always matches.
  Exceptions are propagated to the caller.

  Args:
    policy: The policy being evaluated.
    tool_call: The ToolCall instance.

  Returns:
    True if the predicate matches, False otherwise.
  """
  if policy.when is None:
    return True

  sig = inspect.signature(policy.when)
  params = list(sig.parameters.values())

  if params:
    first_param = params[0]
    # Resolve string annotations if future annotations are active
    try:
      hints = typing.get_type_hints(policy.when)
      annotation = hints.get(first_param.name, first_param.annotation)
    except (TypeError, NameError):
      annotation = first_param.annotation

    if isinstance(annotation, type) and issubclass(
        annotation, pydantic.BaseModel
    ):
      if issubclass(annotation, types.ToolCall):
        raw_result = policy.when(tool_call)
      else:
        typed_args = annotation.model_validate(tool_call.args)
        raw_result = policy.when(typed_args)
    else:
      raw_result = policy.when(tool_call.args)
  else:
    raw_result = policy.when()

  result = await raw_result if inspect.isawaitable(raw_result) else raw_result
  return bool(result)


async def _execute_ask_user(policy: Policy, tool_call: types.ToolCall) -> bool:
  """Invokes the policy's ask_user handler, propagating exceptions."""
  assert policy.ask_user is not None
  result = policy.ask_user(tool_call)
  if inspect.isawaitable(result):
    result = await result
  return bool(result)


def _matches_target(policy_tool: str, tool_call: types.ToolCall) -> bool:
  """Matches a policy tool definition against a tool call target."""
  if policy_tool == _WILDCARD:
    return True
  if tool_call.server_name:
    if policy_tool.endswith("/*"):
      return policy_tool[:-2] == tool_call.server_name
    return policy_tool == f"{tool_call.server_name}/{tool_call.name}"
  return policy_tool == tool_call.name


# ---------------------------------------------------------------------------
# Hook implementation for in-process policy evaluation
# ---------------------------------------------------------------------------


class _PolicyDecideHook(hooks.PreToolCallDecideHook):
  """PreToolCallDecideHook that evaluates dynamic policies in-process."""

  def __init__(self, policies: Sequence[Policy]):
    self._policies = list(policies)

  async def run(
      self, context: hooks.HookContext, data: types.ToolCall
  ) -> hooks.HookResult:
    """Evaluates dynamic policies sequentially against the tool call."""
    del context
    tool_call = data
    for p in self._policies:
      if p.name == _WORKSPACE_ONLY_POLICY_NAME:
        # Workspace boundary containment is enforced at the platform layer.
        continue

      if not _matches_target(p.tool, tool_call):
        continue

      try:
        if not await _evaluate_predicate(p, tool_call):
          continue

        label = p.name or p.tool
        if p.decision == Decision.DENY:
          _logger.info("Policy %r denied tool %r.", label, tool_call.name)
          return hooks.HookResult(
              allow=False,
              message=f"Denied by policy '{label}'.",
          )
        if p.decision == Decision.APPROVE:
          _logger.info("Policy %r approved tool %r.", label, tool_call.name)
          return hooks.HookResult(allow=True)
        if p.decision == Decision.ASK_USER and p.ask_user is not None:
          _logger.info(
              "Policy %r requesting user approval for tool %r.",
              label,
              tool_call.name,
          )
          approved = await _execute_ask_user(p, tool_call)
          if approved:
            return hooks.HookResult(allow=True)
          return hooks.HookResult(
              allow=False,
              message=(
                  f"User denied tool '{tool_call.name}' (policy '{label}')."
              ),
          )
      except Exception as e:  # pylint: disable=broad-exception-caught
        _logger.error(
            "Exception during policy %r evaluation — failing closed.",
            p.name or p.tool,
            exc_info=True,
        )
        return hooks.HookResult(
            allow=False,
            message=(
                f"Policy evaluation failed for policy '{p.name or p.tool}':"
                f" {repr(e)}"
            ),
        )

    return hooks.HookResult(allow=True)


# ---------------------------------------------------------------------------
# Private helpers for hook construction & proto serialization
# ---------------------------------------------------------------------------


def flatten_policies(
    policies: Sequence[Policy | Sequence[Policy]],
) -> list[Policy]:
  """Flattens nested sequences of policies into a flat list.

  This allows combining single Policy objects and lists of Policies (e.g.
  returned by overloaded builders) seamlessly in config declarations.

  Args:
    policies: A sequence of Policy objects or nested sequences of Policies.

  Returns:
    A flat list of Policy objects.
  """
  flat = []
  for p in policies:
    if isinstance(p, Policy):
      flat.append(p)
    elif isinstance(p, Sequence) and not isinstance(p, (str, bytes)):
      for sub_p in p:
        if not isinstance(sub_p, Policy):
          raise ValueError(f"Expected Policy, got {type(sub_p)}")
      flat.extend(p)
    else:
      raise ValueError(
          f"Expected Policy or Sequence of Policies, got {type(p)}"
      )
  return flat


# ---------------------------------------------------------------------------
# Proto serialization
# ---------------------------------------------------------------------------

_DECISION_TO_PROTO = {
    Decision.APPROVE: localharness_pb2.POLICY_DECISION_ALLOW,
    Decision.DENY: localharness_pb2.POLICY_DECISION_DENY,
    Decision.ASK_USER: localharness_pb2.POLICY_DECISION_ASK_USER,
}


# TODO: Add explicit server_name field to Policy dataclass to
# eliminate slashed "server/tool" string packing and remove _parse_tool_target.
def _parse_tool_target(tool: str) -> tuple[str, str]:
  """Decomposes 'server/tool' format into (tool_name, server_name) for proto.

  Args:
    tool: The tool target string (e.g., "run_command", "*", "server/tool",
      "server/*").

  Returns:
    A (tool_name, server_name) tuple for the PolicyRule proto fields.
  """
  if tool == _WILDCARD:
    return ("*", "")
  if "/" in tool:
    server, tool_name = tool.split("/", 1)
    return (tool_name, server)
  return (tool, "")


def _to_policy_config_proto(
    policies: Sequence[Policy | Sequence[Policy]],
) -> tuple[localharness_pb2.PolicyConfig, dict[str, Policy]]:
  """Serializes Python Policy objects into a PolicyConfig proto.

  Static rules (no condition function, not ASK_USER) are handled entirely by
  localharness with zero wire roundtrips. Dynamic rules (with dynamic callbacks
  or ASK_USER handlers) are tagged with a rule_id and require the SDK to
  evaluate the condition when localharness sends a PolicyDecisionRequest.

  Args:
    policies: The policies to serialize (can be nested).

  Returns:
    A tuple of (PolicyConfig proto, dynamic_policy_map). The dynamic_policy_map
    maps rule_id -> Policy for dynamic rules, used by the event processor
    to handle incoming PolicyDecisionRequest messages.
  """
  flat = flatten_policies(policies)

  dynamic_policy_map: dict[str, Policy] = {}
  proto_rules: list[localharness_pb2.PolicyRule] = []

  for i, p in enumerate(flat):
    # TODO: Remove _parse_tool_target once Policy has server_name.
    tool_name, server_name = _parse_tool_target(p.tool)
    is_workspace_only = p.name == _WORKSPACE_ONLY_POLICY_NAME
    is_dynamic = (
        p.when is not None or p.decision == Decision.ASK_USER
    ) and not is_workspace_only

    rule_id = ""
    if is_dynamic:
      rule_id = f"rule_{i}"
      dynamic_policy_map[rule_id] = p

    proto_rules.append(
        localharness_pb2.PolicyRule(
            tool=tool_name,
            server_name=server_name,
            decision=_DECISION_TO_PROTO[p.decision],
            name=p.name or p.tool,
            is_dynamic=is_dynamic,
            rule_id=rule_id,
        )
    )

  config = localharness_pb2.PolicyConfig(
      rules=proto_rules,
  )
  return config, dynamic_policy_map


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def _policy_sort_key(p: Policy) -> tuple[int, int]:
  """Returns a sort key ordering specific before wildcards, and DENY before APPROVE."""
  scope = 2 if p.tool == _WILDCARD else (1 if p.tool.endswith("/*") else 0)
  decision_order = {
      Decision.DENY: 0,
      Decision.ASK_USER: 1,
      Decision.APPROVE: 2,
  }
  return (scope, decision_order.get(p.decision, 3))


def enforce(
    policies: Sequence[Policy | Sequence[Policy]],
    *,
    mcp_servers: Sequence[types.BaseMcpServerConfig] | None = None,
) -> hooks.PreToolCallDecideHook:
  """Creates a PreToolCallDecideHook that evaluates dynamic policies.

  Args:
    policies: The policies to enforce (can be nested).
    mcp_servers: Optional registered MCP server configurations to validate
      against MCP-targeted policies.

  Returns:
    A PreToolCallDecideHook ready for registration.

  Raises:
    ValueError: If any ASK_USER policy is missing a handler.
  """
  flat_policies = sorted(flatten_policies(policies), key=_policy_sort_key)

  # Validate MCP policies against mcp_servers (Fail-Closed Security Guard)
  has_mcp_policy = any(
      ("/" in p.tool and p.tool != _WILDCARD) for p in flat_policies
  )
  if has_mcp_policy and mcp_servers is None:
    raise ValueError(
        "MCP policies (containing '/') were detected, but 'mcp_servers' was not"
        " provided to enforce(). You must pass the registered MCP servers to"
        " enable secure policy matching and prevent silent bypasses."
    )

  # Startup validation.
  for p in flat_policies:
    if p.decision == Decision.ASK_USER and p.ask_user is None:
      raise ValueError(
          f"ASK_USER policy '{p.name or p.tool}' is missing an ask_user"
          " handler. Provide one via policy.ask_user(tool, handler=...)."
      )

  return _PolicyDecideHook(flat_policies)
