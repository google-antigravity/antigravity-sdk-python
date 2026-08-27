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
import urllib.parse
import urllib.request

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
    types.BuiltinTools.START_SUBAGENT: "invoke_subagent",
    types.BuiltinTools.GENERATE_IMAGE: "generate_image",
    types.BuiltinTools.SEARCH_WEB: "search_web",
    types.BuiltinTools.READ_URL_CONTENT: "read_url_content",
    types.BuiltinTools.FINISH: "finish",
}

# Reverse map from canonical StepUpdate proto field name to SDK public
# BuiltinTools value.
PROTO_FIELD_TO_SDK_NAME: dict[str, str] = {
    v: k.value for k, v in BUILTIN_TOOL_PROTO_FIELDS.items()
}

# Argument keys in tool call JSON payloads that carry wire-format URIs
# (file:///..., cns://...) and must be normalized to clean filesystem paths.
WIRE_PATH_ARGUMENT_KEYS: frozenset[str] = frozenset(
    {"path", "file_path", "directory_path", "TargetFile", "output_path"}
)


def normalize_wire_path(path: str) -> str:
  """Translates wire-format URIs to clean absolute filesystem paths.

  Paths arrive as file:/// or cns:// URIs over the wire protocol.
  This converts them to platform-native absolute paths that user code expects.
  """
  parsed = urllib.parse.urlparse(path)
  if parsed.scheme == "file":
    # urlparse("file:///abs/path").path == "/abs/path"
    # url2pathname converts URL path to platform-native path
    return urllib.request.url2pathname(parsed.path)
  if parsed.scheme == "cns":
    # urlparse("cns://el-d/home/user/...").netloc == "el-d"
    # urlparse("cns://el-d/home/user/...").path == "/home/user/..."
    # Convert to the canonical /cns/<cell>/... absolute path format.
    return "/cns/" + parsed.netloc + parsed.path
  return path


def normalize_workspace_path(path: str | os.PathLike[str]) -> str:
  """Normalizes wire URIs, expands user home ~, and resolves relative paths."""
  path_str = os.fspath(path)
  if not path_str:
    return ""
  normalized = normalize_wire_path(path_str)
  if normalized.startswith("/cns/"):
    return normalized
  parsed = urllib.parse.urlparse(normalized)
  if not parsed.scheme or parsed.scheme == "file":
    return str(pathlib.Path(normalized).expanduser().resolve())
  return normalized


def normalize_workspace_paths(
    workspaces: (
        Sequence[str | os.PathLike[str]] | str | os.PathLike[str] | None
    ),
    *,
    default_to_cwd: bool = False,
) -> list[str]:
  """Coerces single/sequence workspace paths and normalizes each entry."""
  if workspaces is None:
    return [os.getcwd()] if default_to_cwd else []
  if isinstance(workspaces, (str, os.PathLike)):
    return [normalize_workspace_path(workspaces)]
  if isinstance(workspaces, (list, tuple, Sequence)) and not isinstance(
      workspaces, (str, bytes)
  ):
    return [normalize_workspace_path(ws) for ws in workspaces]
  raise ValueError(
      f"workspaces must be a sequence of paths, got {type(workspaces).__name__}"
  )


def make_step_id(trajectory_id: str, step_index: int) -> str:
  """Creates a unique step identifier from trajectory ID and step index."""
  return f"{trajectory_id}:{step_index}" if trajectory_id else str(step_index)


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
      retry_config: types.RetryConfig | None = None,
      budget_config: types.BudgetConfig | None = None,
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
        conversation_id=self.conversation_id,
        session_continuation_mode=self.session_continuation_mode,
        save_dir=self._get_or_create_save_dir(),
        workspaces=self.workspaces,
        app_data_dir=self.app_data_dir,
        skills_paths=self.skills_paths,
        mcp_servers=self.mcp_servers,
        env=self.env,
        subagents=self.subagents,
        debug_config=self.debug_config,
        retry_config=self.retry_config,
        budget_config=self.budget_config,
        policies=list(self.policies) if self.policies is not None else None,
        tools=self.tools,
    )
