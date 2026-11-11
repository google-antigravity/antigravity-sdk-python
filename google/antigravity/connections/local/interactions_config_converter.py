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

"""Translates SDK Python configs and client inputs directly to GAOS Interactions JSON."""

import base64
import functools
import inspect
import json
import pathlib
from typing import Any, Callable, Sequence

from google.genai import types as genai_types

from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.hooks import hook_runner as h_runner
from google.antigravity.hooks import policy
from google.antigravity.tools import schema_utils
from google.antigravity.tools import tool_runner as t_runner

_AGENT_BEHAVIOR_JSON_MAP: dict[types.AgentBehavior, str] = {
    types.AgentBehavior.AUTONOMOUS: "autonomous",
    types.AgentBehavior.INTERACTIVE: "interactive",
    types.AgentBehavior.MINIMAL: "minimal",
}

_MODEL_TYPE_JSON_MAP: dict[types.ModelType, str] = {
    types.ModelType.TEXT: "text",
    types.ModelType.IMAGE: "image",
}

_POLICY_DECISION_JSON_MAP: dict[policy.Decision, str] = {
    policy.Decision.APPROVE: "allow",
    policy.Decision.DENY: "deny",
    policy.Decision.ASK_USER: "ask_user",
}

_BUDGET_SCOPE_JSON_MAP: dict[types.BudgetScope, str] = {
    types.BudgetScope.LIFETIME: "lifetime",
    types.BudgetScope.FORWARD_LOOKING: "forward_looking",
}


def _sanitize_prompt(text: str) -> str:
  """Removes null bytes from prompt text."""
  return text.replace("\x00", "")


def callable_to_function_tool_dict(
    fn: Callable[..., Any],
    tool_runner: t_runner.ToolRunner | None = None,
) -> dict[str, Any]:
  """Converts a Python callable directly to a GAOS Interactions function tool dict."""
  if isinstance(fn, t_runner.ToolWithSchema):
    fn_dict: dict[str, Any] = {
        "type": "function",
        "name": getattr(fn, "__name__", ""),
    }
    if fn.__doc__:
      fn_dict["description"] = fn.__doc__
    fn_dict["parameters"] = schema_utils.normalize_schema(fn.input_schema)
    return fn_dict

  target_fn: Callable[..., Any] = fn
  tool_name = t_runner.get_tool_name(fn)
  # When a tool uses ToolContext dependency injection, ToolRunner strips the
  # injected parameter via get_public_callable() so the model's JSON schema only
  # exposes user-facing arguments. tool_runner=None is an optional fallback for
  # standalone callables without injected parameters.
  if tool_runner is not None and tool_name in tool_runner.tools:
    target_fn = tool_runner.get_public_callable(tool_name)

  if not hasattr(target_fn, "__name__"):
    orig_fn = target_fn
    base_fn: Any = orig_fn
    while isinstance(base_fn, functools.partial):
      base_fn = base_fn.func
    routine_fn = (
        base_fn if inspect.isroutine(base_fn) else getattr(base_fn, "__call__")
    )

    @functools.wraps(routine_fn)
    def wrapped() -> None:
      """Introspection stub preserving __wrapped__ for deferred type hints."""

    wrapped.__name__ = tool_name
    wrapped.__doc__ = t_runner.get_tool_doc(fn) or None
    try:
      setattr(wrapped, "__signature__", inspect.signature(orig_fn))
    except (ValueError, TypeError):
      setattr(wrapped, "__signature__", inspect.Signature())
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

  fn_dict = {
      "type": "function",
      "name": decl.name or "",
  }
  if decl.description:
    fn_dict["description"] = decl.description
  fn_dict["parameters"] = schema_utils.normalize_schema(parameters)
  return fn_dict


def _translate_system_instructions(
    instructions: str | types.SystemInstructions | None,
    target: dict[str, Any],
) -> None:
  """Populates developer_instructions or appended_developer_instructions on target."""
  if not instructions:
    return
  if isinstance(instructions, str):
    instructions = types.TemplatedSystemInstructions(
        sections=[types.SystemInstructionSection(content=instructions)]
    )

  if isinstance(instructions, types.CustomSystemInstructions):
    if instructions.text:
      target["developer_instructions"] = [
          {"type": "text", "text": instructions.text}
      ]
  elif isinstance(instructions, types.TemplatedSystemInstructions):
    appended_dict: dict[str, Any] = {}
    if instructions.identity:
      appended_dict["custom_identity"] = instructions.identity
    if instructions.sections:
      appended_dict["appended_sections"] = [
          {"title": sec.title, "content": sec.content}
          for sec in instructions.sections
      ]
    if appended_dict:
      target["appended_developer_instructions"] = appended_dict
  else:
    raise TypeError(
        f"Unsupported system_instructions type: {type(instructions)}"
    )


def _translate_mcp_server(
    server_cfg: types.McpServerConfig,
) -> dict[str, Any]:
  """Converts an SDK McpServerConfig to a GAOS Interactions mcp_server tool dict."""
  mcp_dict: dict[str, Any] = {
      "type": "mcp_server",
      "name": server_cfg.name,
  }
  if server_cfg.timeout_seconds and server_cfg.timeout_seconds > 0:
    mcp_dict["timeout"] = f"{server_cfg.timeout_seconds}s"

  if isinstance(server_cfg, types.McpStdioServer):
    stdio_dict: dict[str, Any] = {"command": server_cfg.command}
    if server_cfg.args:
      stdio_dict["args"] = list(server_cfg.args)
    if server_cfg.env:
      stdio_dict["env"] = dict(server_cfg.env)
    mcp_dict["stdio"] = stdio_dict
  elif isinstance(server_cfg, types.McpStreamableHttpServer):
    http_dict: dict[str, Any] = {"url": server_cfg.url}
    if server_cfg.headers:
      http_dict["headers"] = dict(server_cfg.headers)
    mcp_dict["http"] = http_dict
  else:
    raise ValueError(f"Unknown McpServerConfig type: {type(server_cfg)}")

  if server_cfg.enabled_tools:
    mcp_dict["allowed_tools"] = [
        {"mode": "any", "tools": list(server_cfg.enabled_tools)}
    ]
  elif server_cfg.disabled_tools:
    mcp_dict["allowed_tools"] = [
        {"mode": "none", "tools": list(server_cfg.disabled_tools)}
    ]
  return mcp_dict


def _resolve_active_tools(
    cfg: types.CapabilitiesConfig | types.SubagentCapabilities | None,
    *,
    is_subagent: bool = False,
) -> set[types.BuiltinTools]:
  """Resolves active built-in tools from CapabilitiesConfig or SubagentCapabilities."""
  if cfg is None:
    if is_subagent:
      cfg = types.SubagentCapabilities(
          enabled_tools=types.BuiltinTools.read_only()
      )
    else:
      cfg = types.CapabilitiesConfig()
  return connection.resolve_active_tools(cfg)


def _translate_capabilities_to_tools_and_policy(
    cfg: types.CapabilitiesConfig | types.SubagentCapabilities | None,
    *,
    is_subagent: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Translates SDK capabilities into GAOS built-in tool dicts and AgentPolicy dict."""
  active_tools = _resolve_active_tools(cfg, is_subagent=is_subagent)
  tools_list: list[dict[str, Any]] = []

  run_cmd_cfg = None
  if cfg is not None:
    run_cmd_cfg = getattr(cfg, "run_command_config", None) or getattr(
        cfg, "run_command", None
    )

  if types.BuiltinTools.RUN_COMMAND in active_tools:
    bash_dict: dict[str, Any] = {"type": "bash"}
    if run_cmd_cfg is not None:
      if run_cmd_cfg.timeout_seconds is not None:
        timeout_ms = int(round(run_cmd_cfg.timeout_seconds * 1000))
        if timeout_ms > 0:
          bash_dict["max_timeout_ms"] = timeout_ms
      if run_cmd_cfg.enable_daemons:
        bash_dict["enable_daemon_commands"] = True
      if run_cmd_cfg.enable_sandbox:
        bash_dict["enable_sandbox"] = True
    tools_list.append(bash_dict)

  if (
      types.BuiltinTools.RUN_COMMAND in active_tools
      or types.BuiltinTools.SCHEDULE in active_tools
  ):
    tools_list.append({"type": "manage_task"})

  if types.BuiltinTools.SCHEDULE in active_tools:
    tools_list.append({"type": "schedule"})

  if types.BuiltinTools.SEARCH_WEB in active_tools:
    tools_list.append({"type": "google_search"})

  if types.BuiltinTools.READ_URL_CONTENT in active_tools:
    tools_list.append({"type": "url_context"})

  fs_ops: list[str] = []
  if types.BuiltinTools.VIEW_FILE in active_tools:
    fs_ops.append("file_read")
  if types.BuiltinTools.CREATE_FILE in active_tools:
    fs_ops.append("file_write")
  if types.BuiltinTools.EDIT_FILE in active_tools:
    fs_ops.append("file_edit")
  if types.BuiltinTools.FIND_FILE in active_tools:
    fs_ops.append("file_find")
  if types.BuiltinTools.LIST_DIR in active_tools:
    fs_ops.append("directory_list")
  if types.BuiltinTools.SEARCH_DIR in active_tools:
    fs_ops.append("file_grep")
  if fs_ops:
    tools_list.append({"type": "filesystem", "supported_operations": fs_ops})

  policy_dict: dict[str, Any] = {}
  if types.BuiltinTools.ASK_QUESTION in active_tools:
    policy_dict["enable_user_questions"] = True
  if types.BuiltinTools.GENERATE_IMAGE in active_tools:
    policy_dict["enable_image_generation"] = True

  return tools_list, policy_dict


def _translate_allowed_subagents(
    cfg: types.CapabilitiesConfig | types.SubagentCapabilities | None,
    *,
    is_subagent: bool = False,
) -> tuple[dict[str, Any], int | None, bool]:
  """Translates subagent capabilities into AllowedSubagents dict, max_depth, and enabled flag."""
  active_tools = _resolve_active_tools(cfg, is_subagent=is_subagent)
  subagent_enabled = False
  max_depth: int | None = None
  allowed_names: list[str] = []

  if cfg is not None:
    subagent_enabled = getattr(cfg, "enable_subagents", True) and (
        types.BuiltinTools.INVOKE_SUBAGENT in active_tools
    )
    max_depth = getattr(cfg, "max_subagent_depth", None)
    allowed_names = list(cfg.allowed_subagents or [])
  elif not is_subagent:
    subagent_enabled = types.BuiltinTools.INVOKE_SUBAGENT in active_tools

  if not subagent_enabled:
    allowed_dict: dict[str, Any] = {"disabled": {}}
  elif allowed_names:
    allowed_dict = {"enumerated": {"names": allowed_names}}
  else:
    allowed_dict = {"all": {}}
  return allowed_dict, max_depth, subagent_enabled


def _translate_agent_behavior(behavior: types.AgentBehavior | None) -> str:
  """Translates SDK AgentBehavior enum to GAOS JSON string."""
  if behavior is not None and behavior in _AGENT_BEHAVIOR_JSON_MAP:
    return _AGENT_BEHAVIOR_JSON_MAP[behavior]
  return "autonomous"


def _translate_subagent_skills_config(
    skills_config: (
        types.SubagentSkillsConfig
        | types.SubagentInheritSkillsConfig
        | types.SubagentNoneSkillsConfig
        | types.SubagentOverrideSkillsConfig
        | None
    ),
) -> dict[str, Any] | None:
  """Translates SubagentSkillsConfig into a GAOS SubagentSkillsConfig dict."""
  if skills_config is None:
    return None
  if isinstance(skills_config, types.SubagentInheritSkillsConfig):
    skills_config = types.SubagentSkillsConfig(inherit_config=skills_config)
  elif isinstance(skills_config, types.SubagentNoneSkillsConfig):
    skills_config = types.SubagentSkillsConfig(none_config=skills_config)
  elif isinstance(skills_config, types.SubagentOverrideSkillsConfig):
    skills_config = types.SubagentSkillsConfig(override_config=skills_config)

  if skills_config.none_config is not None:
    return {"none_config": {}}
  if skills_config.inherit_config is not None:
    inherit_dict: dict[str, Any] = {}
    if skills_config.inherit_config.skill_names:
      inherit_dict["skill_names"] = list(
          skills_config.inherit_config.skill_names
      )
    if skills_config.inherit_config.extra_skills_paths:
      inherit_dict["extra_skills_paths"] = list(
          skills_config.inherit_config.extra_skills_paths
      )
    return {"inherit_config": inherit_dict}
  if skills_config.override_config is not None:
    if skills_config.override_config.inline_skills:
      raise ValueError(
          "inline_skills in SubagentOverrideSkillsConfig is not supported"
          " by LocalConnectionStrategy; use AntigravityProdActorAgentConfig or"
          " skills_paths instead."
      )
    override_dict: dict[str, Any] = {}
    if skills_config.override_config.skills_paths:
      override_dict["skills_paths"] = list(
          skills_config.override_config.skills_paths
      )
    return {"override_config": override_dict}
  return None


def _translate_custom_subagent(
    subagent: types.SubagentConfig,
    all_tool_dicts: dict[str, dict[str, Any]],
    tool_runner: t_runner.ToolRunner | None = None,
) -> dict[str, Any]:
  """Translates an SDK SubagentConfig to an AntigravitySubagentsConfig.CustomAgent dict."""
  capabilities = subagent.capabilities or types.SubagentCapabilities(
      enabled_tools=types.BuiltinTools.read_only(),
  )
  sa_dict: dict[str, Any] = {"name": subagent.name}
  if subagent.description:
    sa_dict["description"] = subagent.description
  if subagent.model is not None:
    sa_dict["model"] = {"name": subagent.model}
  _translate_system_instructions(subagent.system_instructions, sa_dict)

  resolved_subagent_tools: list[dict[str, Any]] = []
  for tool in subagent.tools or []:
    if isinstance(tool, str):
      if tool in all_tool_dicts:
        resolved_subagent_tools.append(all_tool_dicts[tool])
      else:
        resolved_subagent_tools.append({
            "type": "function",
            "name": tool,
            "parameters": {"type": "object", "properties": {}},
        })
    elif callable(tool):
      fn_dict = callable_to_function_tool_dict(tool, tool_runner=tool_runner)
      all_tool_dicts[fn_dict["name"]] = fn_dict
      resolved_subagent_tools.append(fn_dict)
    else:
      raise ValueError(
          f"Invalid tool type in subagent '{subagent.name}' tools list: {tool}"
      )

  cap_tools, policy_dict = _translate_capabilities_to_tools_and_policy(
      capabilities, is_subagent=True
  )
  tools_list = resolved_subagent_tools + cap_tools
  if tools_list:
    sa_dict["tools"] = tools_list
  if policy_dict:
    sa_dict["policy"] = policy_dict

  allowed_dict, _, _ = _translate_allowed_subagents(
      capabilities, is_subagent=True
  )
  sa_dict["allowed_subagents"] = allowed_dict
  sa_dict["agent_behavior"] = _translate_agent_behavior(
      capabilities.agent_behavior
  )
  skills_cfg_dict = _translate_subagent_skills_config(subagent.skills_config)
  if skills_cfg_dict is not None:
    sa_dict["skills_config"] = skills_cfg_dict
  return sa_dict


def _translate_gemini_options(
    options: types.GeminiModelOptions | None,
) -> dict[str, Any]:
  """Translates SDK GeminiModelOptions to a GAOS options dict."""
  opts: dict[str, Any] = {}
  if options:
    if options.thinking_level:
      opts["thinking_level"] = str(options.thinking_level.value)
    if options.service_tier:
      if hasattr(options.service_tier, "value"):
        opts["service_tier"] = str(options.service_tier.value)
      else:
        opts["service_tier"] = str(options.service_tier)
  return opts


def _translate_model_target(m: types.ModelTarget) -> dict[str, Any]:
  """Translates an SDK ModelTarget to an AntigravityAgentConfig.ModelConfig dict."""
  m_dict: dict[str, Any] = {"name": m.name or ""}
  model_types = [
      _MODEL_TYPE_JSON_MAP[t] for t in m.types if t in _MODEL_TYPE_JSON_MAP
  ]
  if model_types:
    m_dict["types"] = model_types

  if isinstance(m.endpoint, types.GeminiAPIEndpoint):
    ep_dict: dict[str, Any] = {}
    if m.endpoint.base_url:
      ep_dict["base_url"] = m.endpoint.base_url
    if m.endpoint.http_headers:
      ep_dict["http_headers"] = dict(m.endpoint.http_headers)
    if m.endpoint.api_key:
      ep_dict["api_key"] = m.endpoint.api_key
    opts = _translate_gemini_options(m.endpoint.options)
    if opts:
      ep_dict["options"] = opts
    m_dict["gemini_api_endpoint"] = ep_dict
  elif isinstance(m.endpoint, types.VertexEndpoint):
    ep_dict = {}
    if m.endpoint.base_url:
      ep_dict["base_url"] = m.endpoint.base_url
    if m.endpoint.http_headers:
      ep_dict["http_headers"] = dict(m.endpoint.http_headers)
    if m.endpoint.project:
      ep_dict["project"] = m.endpoint.project
    if m.endpoint.location:
      ep_dict["location"] = m.endpoint.location
    if m.endpoint.api_key:
      ep_dict["api_key"] = m.endpoint.api_key
    opts = _translate_gemini_options(m.endpoint.options)
    if opts:
      ep_dict["options"] = opts
    m_dict["vertex_endpoint"] = ep_dict
  else:
    raise ValueError(f"Unrecognized endpoint type: {type(m.endpoint)}")
  return m_dict


def _get_enabled_hooks(hook_runner: h_runner.HookRunner | None) -> list[str]:
  """Returns the GAOS lifecycle hook names registered on hook_runner."""
  if not hook_runner:
    return []
  hook_mapping = [
      (hook_runner.on_session_start_hooks, "on_session_start"),
      (hook_runner.on_session_end_hooks, "on_session_end"),
      (hook_runner.pre_turn_hooks, "pre_turn"),
      (hook_runner.post_turn_hooks, "post_turn"),
      (hook_runner.pre_tool_call_decide_hooks, "pre_tool"),
      (hook_runner.post_tool_call_hooks, "post_tool"),
      (hook_runner.on_tool_error_hooks, "on_tool_error"),
      (hook_runner.on_compaction_hooks, "on_compaction"),
      (hook_runner.stop_hooks, "stop"),
  ]
  return [name for hooks_list, name in hook_mapping if hooks_list]


def _translate_policy_config(
    policies: Sequence[policy.Policy | Sequence[policy.Policy]],
) -> dict[str, Any]:
  """Translates SDK Policy objects directly to GAOS AntigravityAgentConfig.PolicyConfig.

  Fails closed with AntigravityValidationError if AutoPolicy or dynamic policy
  rules (with `when` predicates or `ask_user` handlers) are present, as those
  are not supported over the GAOS Interactions API in localharness.

  Args:
    policies: Sequence of SDK Policy objects (or nested sequences).

  Returns:
    A dictionary representing GAOS PolicyConfig.
  """
  flat = policy.flatten_policies(policies)
  rules_list: list[dict[str, Any]] = []
  has_workspace_only = False
  has_allow_all = False

  for p in flat:
    if p.auto:
      raise types.AntigravityValidationError(
          "policy.auto() is not yet supported by the GAOS Interactions API"
          " protocol in localharness."
      )
    is_workspace_only = p.name == policy.WORKSPACE_ONLY_POLICY_NAME
    if is_workspace_only:
      has_workspace_only = True
    if (
        p.name == "allow_all"
        and p.tool == "*"
        and p.decision == policy.Decision.APPROVE
        and p.when is None
    ):
      has_allow_all = True

    is_dynamic = (
        p.when is not None or p.decision == policy.Decision.ASK_USER
    ) and not is_workspace_only
    if is_dynamic:
      raise types.AntigravityValidationError(
          "Dynamic policy rules (with 'when=' predicates or 'ask_user'"
          f" handlers, such as '{p.name or p.tool}') are not yet supported by"
          " the GAOS Interactions API protocol in localharness."
      )

    if p.tool == "*":
      tool_name, server_name = "*", ""
    elif "/" in p.tool:
      server_name, tool_name = p.tool.split("/", 1)
    else:
      tool_name, server_name = p.tool, ""

    r_dict: dict[str, Any] = {}
    if tool_name:
      r_dict["tool"] = tool_name
    if server_name:
      r_dict["server_name"] = server_name
    r_dict["name"] = p.name or p.tool
    if p.decision in _POLICY_DECISION_JSON_MAP:
      r_dict["decision"] = _POLICY_DECISION_JSON_MAP[p.decision]
    if p.reason:
      r_dict["deny_reason"] = p.reason
    rules_list.append(r_dict)

  pc_dict: dict[str, Any] = {}
  if rules_list:
    pc_dict["rules"] = rules_list
  if has_allow_all and not has_workspace_only:
    pc_dict["workspace_containment"] = "disabled"
  return pc_dict


def _translate_budget_config(
    budget_config: types.BudgetConfig,
) -> dict[str, Any]:
  """Translates SDK BudgetConfig to GAOS AntigravityAgentConfig.BudgetConfig dict."""
  bc_dict: dict[str, Any] = {}
  if budget_config.max_model_calls and budget_config.max_model_calls > 0:
    bc_dict["max_model_calls"] = budget_config.max_model_calls
  if budget_config.max_tool_calls and budget_config.max_tool_calls > 0:
    bc_dict["max_tool_calls"] = budget_config.max_tool_calls
  if budget_config.max_input_tokens and budget_config.max_input_tokens > 0:
    bc_dict["max_input_tokens"] = budget_config.max_input_tokens
  if budget_config.max_output_tokens and budget_config.max_output_tokens > 0:
    bc_dict["max_output_tokens"] = budget_config.max_output_tokens
  if budget_config.max_total_tokens and budget_config.max_total_tokens > 0:
    bc_dict["max_total_tokens"] = budget_config.max_total_tokens
  if not bc_dict:
    return {}
  if budget_config.scope in _BUDGET_SCOPE_JSON_MAP:
    bc_dict["scope"] = _BUDGET_SCOPE_JSON_MAP[budget_config.scope]
  return bc_dict


def _translate_retry_config(
    retry_config: types.RetryConfig,
) -> dict[str, Any]:
  """Translates SDK RetryConfig to GAOS AntigravityAgentConfig.RetryConfig dict."""
  rc_dict: dict[str, Any] = {}
  if retry_config.api_retry:
    api_data = retry_config.api_retry.model_dump(exclude_none=True)
    if api_data:
      rc_dict["api_retry"] = api_data
  if retry_config.model_output_retry:
    out_data = retry_config.model_output_retry.model_dump(exclude_none=True)
    if out_data:
      rc_dict["model_output_retry"] = out_data
  return rc_dict


def build_create_interaction_event(
    *,
    models: Sequence[types.ModelTarget] | None = None,
    system_instructions: str | types.SystemInstructions | None = None,
    capabilities_config: types.CapabilitiesConfig | None = None,
    compaction_config: types.CompactionConfig | None = None,
    conversation_id: str | None = None,
    session_continuation_mode: types.SessionContinuationMode | None = None,
    workspaces: Sequence[str] | None = None,
    skills_paths: Sequence[str] | None = None,
    app_data_dir: str | None = None,
    mcp_servers: Sequence[types.McpServerConfig] | None = None,
    subagents: Sequence[types.SubagentConfig] | None = None,
    retry_config: types.RetryConfig | None = None,
    budget_config: types.BudgetConfig | None = None,
    policies: Sequence[policy.Policy | Sequence[policy.Policy]] | None = None,
    tools: Sequence[Callable[..., Any] | str] | None = None,
    tool_runner: t_runner.ToolRunner | None = None,
    hook_runner: h_runner.HookRunner | None = None,
    initial_trajectory: bytes | None = None,
) -> dict[str, Any]:
  """Builds a GAOS CreateInteractionEvent dict directly from SDK Python configs."""
  event: dict[str, Any] = {
      "event_type": "interaction.create",
      "agent": "antigravity",
  }

  if conversation_id:
    if session_continuation_mode == types.SessionContinuationMode.CREATE_ONLY:
      event["interaction_id"] = conversation_id
    elif session_continuation_mode == types.SessionContinuationMode.RESUME:
      event["previous_interaction_id"] = conversation_id
    else:
      event["interaction_id"] = conversation_id
      event["previous_interaction_id"] = conversation_id

  _translate_system_instructions(system_instructions, event)

  effective_capabilities = capabilities_config or types.CapabilitiesConfig()

  all_tool_dicts: dict[str, dict[str, Any]] = {}
  if tool_runner:
    for fn in tool_runner.tools.values():
      fn_dict = callable_to_function_tool_dict(fn, tool_runner=tool_runner)
      all_tool_dicts[fn_dict["name"]] = fn_dict

  root_tool_dicts: list[dict[str, Any]] = []
  if tools is not None:
    for tool in tools:
      if isinstance(tool, str):
        if tool in all_tool_dicts:
          root_tool_dicts.append(all_tool_dicts[tool])
        else:
          root_tool_dicts.append({
              "type": "function",
              "name": tool,
              "parameters": {"type": "object", "properties": {}},
          })
      elif callable(tool):
        fn_dict = callable_to_function_tool_dict(tool, tool_runner=tool_runner)
        all_tool_dicts[fn_dict["name"]] = fn_dict
        root_tool_dicts.append(fn_dict)
  elif tool_runner:
    subagent_tool_names: set[str] = set()
    for sa in subagents or []:
      for t in sa.tools or []:
        if isinstance(t, str):
          subagent_tool_names.add(t)
        elif callable(t):
          subagent_fn = callable_to_function_tool_dict(
              t, tool_runner=tool_runner
          )
          subagent_tool_names.add(subagent_fn["name"])
    root_tool_dicts = [
        fn_dict
        for name, fn_dict in all_tool_dicts.items()
        if name not in subagent_tool_names
    ]

  mcp_tool_dicts = [_translate_mcp_server(s) for s in mcp_servers or []]
  cap_tool_dicts, policy_dict = _translate_capabilities_to_tools_and_policy(
      effective_capabilities, is_subagent=False
  )

  tools_list = root_tool_dicts + mcp_tool_dicts + cap_tool_dicts
  if tools_list:
    event["tools"] = tools_list

  agent_config: dict[str, Any] = {"type": "antigravity"}

  if policy_dict:
    agent_config["policy"] = policy_dict

  allowed_subagents_dict, max_depth, subagent_enabled = (
      _translate_allowed_subagents(effective_capabilities, is_subagent=False)
  )
  if subagent_enabled or subagents:
    subagents_cfg: dict[str, Any] = {
        "allowed_subagents": allowed_subagents_dict,
    }
    if max_depth is not None and max_depth > 0:
      subagents_cfg["max_nesting_depth"] = max_depth
    if subagents:
      subagents_cfg["custom_subagents"] = [
          _translate_custom_subagent(
              sa, all_tool_dicts, tool_runner=tool_runner
          )
          for sa in subagents
      ]
    agent_config["subagents_config"] = subagents_cfg

  if models:
    agent_config["models"] = {
        "models": [_translate_model_target(m) for m in models]
    }

  if workspaces:
    agent_config["workspaces"] = [
        {"filesystem_workspace": {"directory": pathlib.Path(p).as_posix()}}
        for p in workspaces
    ]

  if skills_paths:
    agent_config["skills_paths"] = list(skills_paths)

  if app_data_dir:
    agent_config["app_data_dir"] = app_data_dir

  enabled_hooks = _get_enabled_hooks(hook_runner)
  if enabled_hooks:
    agent_config["enabled_hooks"] = enabled_hooks

  effective_compaction = compaction_config
  if effective_compaction is None:
    raw_threshold = (
        effective_capabilities._get_explicit_compaction_threshold()  # pylint: disable=protected-access
    )
    if raw_threshold is not None:
      effective_compaction = types.CompactionConfig(
          token_threshold=raw_threshold,
      )
  if (
      effective_compaction is not None
      and effective_compaction.token_threshold
      and effective_compaction.token_threshold > 0
  ):
    agent_config["compaction_config"] = {
        "token_threshold": effective_compaction.token_threshold
    }

  if effective_capabilities.finish_tool_schema_json:
    agent_config["finish_tool_output_schema"] = json.loads(
        effective_capabilities.finish_tool_schema_json
    )

  truncation_config = effective_capabilities.tool_output_truncation_config
  if truncation_config is not None:
    agent_config["tool_output_truncation"] = {
        "truncate": {"max_tokens": truncation_config.max_tokens}
    }

  if retry_config is not None:
    rc_dict = _translate_retry_config(retry_config)
    if rc_dict:
      agent_config["retry_config"] = rc_dict

  if policies:
    pc_dict = _translate_policy_config(policies)
    if pc_dict:
      agent_config["policy_config"] = pc_dict

  if budget_config is not None:
    bc_dict = _translate_budget_config(budget_config)
    if bc_dict:
      agent_config["budget_config"] = bc_dict

  agent_config["agent_behavior"] = _translate_agent_behavior(
      effective_capabilities.agent_behavior
  )

  if initial_trajectory:
    agent_config["initial_trajectory"] = base64.b64encode(
        initial_trajectory
    ).decode("ascii")

  event["agent_config"] = agent_config
  return event


def _primitive_to_content_dict(item: types.ContentPrimitive) -> dict[str, Any]:
  """Converts a single non-slash-command ContentPrimitive into a GAOS Content dict."""
  if isinstance(item, str):
    return {"type": "text", "text": _sanitize_prompt(item)}
  if isinstance(item, types.Image):
    return {
        "type": "image",
        "mime_type": item.mime_type,
        "data": base64.b64encode(item.data).decode("ascii"),
    }
  if isinstance(item, types.Audio):
    return {
        "type": "audio",
        "mime_type": item.mime_type,
        "data": base64.b64encode(item.data).decode("ascii"),
    }
  if isinstance(item, types.Video):
    return {
        "type": "video",
        "mime_type": item.mime_type,
        "data": base64.b64encode(item.data).decode("ascii"),
    }
  if isinstance(item, types.Document):
    return {
        "type": "document",
        "mime_type": item.mime_type,
        "data": base64.b64encode(item.data).decode("ascii"),
    }
  raise TypeError(f"Unsupported prompt content type: {type(item)}")


def _slash_command_name(sc: types.SlashCommand) -> str:
  """Returns the string value of a SlashCommand name."""
  return sc.name.value if hasattr(sc.name, "value") else str(sc.name)


def _primitive_to_input_content_dict(
    item: types.ContentPrimitive,
) -> dict[str, Any]:
  """Converts an SDK primitive into a GAOS UserInputContent dict."""
  if isinstance(item, types.SlashCommand):
    return {"type": "slash_command", "name": _slash_command_name(item)}
  return _primitive_to_content_dict(item)


def content_to_user_input_event(content: types.Content) -> dict[str, Any]:
  """Converts SDK Content into a GAOS ExternalClientEvent (InputEvent)."""
  items = (
      [content]
      if isinstance(content, str) or not isinstance(content, Sequence)
      else list(content)
  )
  if not any(isinstance(item, types.SlashCommand) for item in items):
    return {
        "event_type": "input",
        "content": [_primitive_to_content_dict(item) for item in items],
    }
  # Slash commands can only be expressed in input_content, where they may
  # be interleaved with other content.
  return {
      "event_type": "input",
      "input_content": [
          _primitive_to_input_content_dict(item) for item in items
      ],
  }


def tool_result_to_function_result_event(
    call_id: str,
    tool_name: str,
    result_dict: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
  """Builds a GAOS ExternalClientEvent (FunctionResultEvent)."""
  event_dict: dict[str, Any] = {
      "event_type": "function_result",
      "call_id": call_id,
      "name": tool_name,
  }
  if error_message is not None:
    event_dict["is_error"] = True
    event_dict["result"] = error_message
  else:
    event_dict["is_error"] = False
    event_dict["result"] = result_dict if result_dict is not None else {}
  return event_dict


def question_responses_to_elicitation_result_events(
    elicitation_ids: Sequence[str],
    responses: Sequence[types.QuestionResponse] | None = None,
    *,
    cancelled: bool = False,
) -> list[dict[str, Any]]:
  """Builds GAOS ElicitationResultEvent dicts for a set of question elicitation IDs."""
  if cancelled:
    if not elicitation_ids:
      return []
    return [{
        "event_type": "elicitation_result",
        "elicitation_id": elicitation_ids[0],
        "confirmation": {"is_confirmed": False},
    }]

  events: list[dict[str, Any]] = []
  for i, el_id in enumerate(elicitation_ids):
    resp = responses[i] if responses and i < len(responses) else None
    mc_dict: dict[str, Any] = {}
    if resp is not None and not resp.skipped:
      if resp.selected_option_ids:
        mc_dict["selected_choice_labels"] = list(resp.selected_option_ids)
      if resp.freeform_response:
        mc_dict["user_input"] = [
            {"type": "text", "text": resp.freeform_response}
        ]
    events.append({
        "event_type": "elicitation_result",
        "elicitation_id": el_id,
        "multiple_choice": mc_dict,
    })
  return events


def build_cancel_interaction_event() -> dict[str, Any]:
  """Builds a GAOS ExternalClientEvent to cancel/halt the current turn."""
  return {"event_type": "interaction.cancel"}


def build_complete_interaction_event() -> dict[str, Any]:
  """Builds a GAOS ExternalClientEvent to complete/end the session."""
  return {"event_type": "interaction.complete"}
