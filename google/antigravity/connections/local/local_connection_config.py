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

"""Configuration for the local harness connection strategy."""

import logging
import os
import pathlib
import tempfile
from typing import Any, Callable, Sequence

import pydantic

from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.hooks import hooks as hooks_mod
from google.antigravity.hooks import policy
from google.antigravity.models import DEFAULT_IMAGE_GENERATION_MODEL
from google.antigravity.models import DEFAULT_MODEL
from google.antigravity.triggers import triggers as triggers_mod

DEFAULT_APP_DATA_DIR = (
    (pathlib.Path("~") / ".gemini" / "antigravity").expanduser().resolve()
)

# Map from BuiltinTools enum to the canonical proto field name on StepUpdate.
BUILTIN_TOOL_PROTO_FIELDS: dict[types.BuiltinTools, str] = {
    types.BuiltinTools.CREATE_FILE: "create_file",
    types.BuiltinTools.EDIT_FILE: "edit_file",
    types.BuiltinTools.FIND_FILE: "find_file",
    types.BuiltinTools.LIST_DIR: "list_directory",
    types.BuiltinTools.RUN_COMMAND: "run_command",
    types.BuiltinTools.SEARCH_DIR: "search_directory",
    types.BuiltinTools.VIEW_FILE: "view_file",
    types.BuiltinTools.INVOKE_SUBAGENT: "invoke_subagent",
    types.BuiltinTools.RUN_WORKFLOW: "run_workflow",
    types.BuiltinTools.GENERATE_IMAGE: "generate_image",
    types.BuiltinTools.SEARCH_WEB: "search_web",
    types.BuiltinTools.READ_URL_CONTENT: "read_url_content",
    types.BuiltinTools.FINISH: "finish",
}

# Reverse map from canonical StepUpdate proto field name to SDK tool name
# (including harness-internal actions like skill_lookup that are not exposed
# on the public BuiltinTools enum).
PROTO_FIELD_TO_SDK_NAME: dict[str, str] = {
    **{v: k.value for k, v in BUILTIN_TOOL_PROTO_FIELDS.items()},
    "skill_lookup": "skill_lookup",
}

# Argument keys in tool call JSON payloads that carry wire-format URIs
# (file:///..., cns://...) and must be normalized to clean filesystem paths.
# Stored as a tuple with fixed priority order so canonical path resolution is
# deterministic.
WIRE_PATH_ARGUMENT_KEYS: tuple[str, ...] = (
    "path",
    "file_path",
    "directory_path",
    "TargetFile",
    "output_path",
    "script_path",
    "ScriptPath",
)

normalize_wire_path = connection.normalize_wire_path
normalize_workspace_path = connection.normalize_workspace_path
normalize_workspace_paths = connection.normalize_workspace_paths
make_step_id = connection.make_step_id


def normalize_path_arguments(args: dict[str, Any]) -> str | None:
  """Normalizes wire-format URIs to clean filesystem paths in-place.

  Paths arrive as file:/// or cns:// URIs over the wire protocol. User hooks and
  policy checks expect clean absolute paths, so we normalize known path fields
  before dispatch.

  Args:
    args: The tool call arguments dictionary to normalize in-place.

  Returns:
    The canonical path (the first normalized path field found), or None.
  """
  canonical_path: str | None = None
  for key in WIRE_PATH_ARGUMENT_KEYS:
    val = args.get(key)
    if isinstance(val, str) and val:
      normalized = normalize_wire_path(val)
      args[key] = normalized
      if canonical_path is None:
        canonical_path = normalized
  return canonical_path


class BaseLocalAgentConfig(connection.AgentConfig):
  """Base configuration class for local harness agent configurations."""

  model_config = pydantic.ConfigDict(
      arbitrary_types_allowed=True, validate_assignment=True
  )

  capabilities: types.CapabilitiesConfig = pydantic.Field(
      default_factory=types.CapabilitiesConfig
  )
  policies: list[policy.Policy] = pydantic.Field(
      default_factory=policy.confirm_run_command
  )
  env: dict[str, str] | None = pydantic.Field(
      default=None,
      description=(
          "Optional custom environment variables for the localharness"
          " subprocess."
      ),
  )
  workspaces: list[str] = pydantic.Field(default_factory=lambda: [os.getcwd()])

  @pydantic.field_validator("workspaces", mode="before")
  @classmethod
  def _validate_workspaces(cls, v: Any) -> list[str]:
    return normalize_workspace_paths(v, default_to_cwd=True)

  @pydantic.field_validator("app_data_dir")
  @classmethod
  def _validate_app_data_dir(cls, v: str | None) -> str | None:
    if v is not None and not os.path.isabs(v):
      raise ValueError(f"app_data_dir must be an absolute path, got '{v}'")
    return v

  @pydantic.model_validator(mode="after")
  def _validate_allowed_subagents(self) -> "BaseLocalAgentConfig":
    declared_names = {
        sub.name for sub in self.subagents or [] if getattr(sub, "name", None)
    }
    valid_names = declared_names
    if self.capabilities and self.capabilities.allowed_subagents is not None:
      unknown = set(self.capabilities.allowed_subagents) - valid_names
      if unknown:
        raise ValueError(
            "Unknown subagent name(s) in CapabilitiesConfig.allowed_subagents:"
            f" {sorted(unknown)}. Valid subagents are: {sorted(valid_names)}"
        )
    for sub in self.subagents or []:
      if sub.capabilities and sub.capabilities.allowed_subagents is not None:
        unknown = set(sub.capabilities.allowed_subagents) - valid_names
        if unknown:
          raise ValueError(
              "Unknown subagent name(s) in"
              f" SubagentConfig('{sub.name}').capabilities.allowed_subagents:"
              f" {sorted(unknown)}. Valid subagents are: {sorted(valid_names)}"
          )
    return self

  @pydantic.model_validator(mode="after")
  def _validate_skills_sources(self) -> "BaseLocalAgentConfig":
    if self.skills_paths and self.inline_skills:
      raise ValueError(
          "combining inline_skills and skills_paths is not supported by"
          " LocalHarness; provide either inline_skills or skills_paths, not"
          " both."
      )
    return self

  def _get_system_instructions(self) -> types.SystemInstructions | None:
    """Returns the system instructions, normalizing shorthand if needed."""
    if isinstance(self.system_instructions, str):
      return types.TemplatedSystemInstructions(
          sections=[
              types.SystemInstructionSection(content=self.system_instructions)
          ]
      )
    return self.system_instructions

  def _get_or_create_save_dir(self) -> str:
    """Returns save_dir, generating a temporary one if not specified."""
    save_dir = self.save_dir
    if save_dir is None:
      save_dir = tempfile.mkdtemp(prefix="antigravity_")
      logging.info("No save_dir specified; using %s", save_dir)
    return save_dir


class LocalAgentConfig(BaseLocalAgentConfig):
  """Configuration for the local harness backend.

  This is the default config for the Agent class. It uses the
  Go-based localharness binary.

  By default, all tools are enabled but ``run_command`` is denied via
  ``policy.confirm_run_command()``.  To enable fully autonomous execution
  (including shell access), pass ``policies=[policy.allow_all()]``.

  When ``workspaces`` are configured, file tools are automatically
  restricted to those directories by the policy evaluator.
  """

  # Top-level shorthand fields — flow into models.
  model: str | types.ModelTarget | None = None
  models: list[types.ModelTarget] | None = None
  api_key: str | None = None
  vertex: bool | None = None
  project: str | None = None
  location: str | None = None

  def __init__(  # pylint: disable=super-init-not-called
      self,
      *,
      system_instructions: str | types.SystemInstructions | None = None,
      capabilities: types.CapabilitiesConfig | None = None,
      tools: list[Callable[..., Any]] | None = None,
      policies: Sequence[policy.Policy | Sequence[policy.Policy]] | None = None,
      hooks: list[hooks_mod.Hook] | None = None,
      triggers: list[triggers_mod.Trigger] | None = None,
      mcp_servers: list[types.McpServerConfig] | None = None,
      subagents: list[types.SubagentConfig] | None = None,
      workspaces: list[str] | None = None,
      env: dict[str, str] | None = None,
      conversation_id: str | None = None,
      session_continuation_mode: types.SessionContinuationMode | None = None,
      save_dir: str | None = None,
      app_data_dir: str | None = None,
      response_schema: (
          dict[str, Any] | type[pydantic.BaseModel] | str | None
      ) = None,
      skills_paths: list[str] | None = None,
      inline_skills: list[types.InlineSkill] | None = None,
      retry_config: types.RetryConfig | None = None,
      budget_config: types.BudgetConfig | None = None,
      compaction_config: types.CompactionConfig | None = None,
      model: str | types.ModelTarget | None = None,
      models: list[types.ModelTarget] | None = None,
      api_key: str | None = None,
      vertex: bool | None = None,
      project: str | None = None,
      location: str | None = None,
      **kwargs: Any,
  ):
    if vertex is None and (
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")
        or os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "").lower()
        in ("true", "1")
    ):
      vertex = True

    init_data = {
        k: v
        for k, v in locals().items()
        if k not in ("self", "init_data") and v is not None
    }
    if "kwargs" in init_data:
      kwargs_dict = init_data.pop("kwargs")
      if isinstance(kwargs_dict, dict):
        init_data.update(
            {k: v for k, v in kwargs_dict.items() if v is not None}
        )
    pydantic.BaseModel.__init__(self, **init_data)

  def _build_shorthand_endpoint(self) -> types.ModelEndpoint | None:
    """Builds the custom endpoint from connection shorthand fields."""
    if self.vertex:
      return types.VertexEndpoint(
          project=self.project,
          location=self.location,
          api_key=self.api_key,
      )
    return types.GeminiAPIEndpoint(api_key=self.api_key)

  def _build_shorthand_models(
      self, endpoint: types.ModelEndpoint | None
  ) -> list[types.ModelTarget]:
    """Builds the explicitly-specified shorthand models."""
    if self.model is None:
      return []

    if isinstance(self.model, types.ModelTarget):
      shorthand_model = self.model.model_copy(deep=True)
      if shorthand_model.endpoint is None:
        shorthand_model.endpoint = endpoint
    else:
      shorthand_model = types.ModelTarget(
          name=self.model, types=[types.ModelType.TEXT], endpoint=endpoint
      )
    return [shorthand_model]

  def _build_default_models(
      self, endpoint: types.ModelEndpoint | None
  ) -> list[types.ModelTarget]:
    """Builds the default text and image models."""
    text_model = types.ModelTarget(
        name=DEFAULT_MODEL,
        types=[types.ModelType.TEXT],
        endpoint=endpoint,
    )
    image_model = types.ModelTarget(
        name=DEFAULT_IMAGE_GENERATION_MODEL,
        types=[types.ModelType.IMAGE],
        endpoint=endpoint,
    )
    return [text_model, image_model]

  def _merge_models_list(self) -> list[types.ModelTarget]:
    """Merges explicit, shorthand, and default models based on priority.

    Priority order is: Explicit > Shorthand > Default.
    Default models are only added if their types are not already present in the
    collection.

    Returns:
      The merged list of model targets.
    """
    endpoint = self._build_shorthand_endpoint()
    explicit_models = self.models or []
    shorthand_models = self._build_shorthand_models(endpoint)
    default_models = self._build_default_models(endpoint)

    merged_models = list(explicit_models)
    for model in shorthand_models:
      merged_models.append(model)

    existing_types = set()
    for m in merged_models:
      existing_types.update(m.types)

    for default_model in default_models:
      if not any(t in existing_types for t in default_model.types):
        merged_models.append(default_model)

    return merged_models

  @pydantic.model_validator(mode="after")
  def _apply_shorthand_configs(self) -> "LocalAgentConfig":
    """Applies top-level shorthand fields (model, api_key) to models."""
    self.__dict__["models"] = self._merge_models_list()
    return self

  def create_strategy(
      self,
      *,
      tool_runner: Any,
      hook_runner: Any,
  ) -> "connection.ConnectionStrategy":
    # Late import to avoid circular dependency: local_connection.py imports
    # this config module, so we import the strategy class here at call time.
    from google.antigravity.connections.local import local_connection  # pylint: disable=g-import-not-at-top

    return local_connection.LocalConnectionStrategy(
        tool_runner=tool_runner,
        hook_runner=hook_runner,
        models=self.models,
        system_instructions=self._get_system_instructions(),
        capabilities_config=self.capabilities,
        compaction_config=self._get_effective_compaction_config(),
        conversation_id=self.conversation_id,
        session_continuation_mode=self.session_continuation_mode,
        save_dir=self._get_or_create_save_dir(),
        workspaces=self.workspaces,
        app_data_dir=self.app_data_dir,
        skills_paths=self.skills_paths,
        inline_skills=self.inline_skills,
        mcp_servers=self.mcp_servers,
        env=self.env,
        subagents=self.subagents,
        debug_config=self.debug_config,
        retry_config=self.retry_config,
        budget_config=self.budget_config,
        policies=list(self.policies) if self.policies is not None else None,
        tools=self.tools,
    )
