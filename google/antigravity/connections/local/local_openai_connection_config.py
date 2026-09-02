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

"""Local OpenAI Agent Config."""

from typing import Any, Callable

import pydantic

from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.connections.local.local_connection_config import BaseLocalAgentConfig
from google.antigravity.hooks import hooks as hooks_mod
from google.antigravity.hooks import policy
from google.antigravity.triggers import triggers as triggers_mod


class LocalOpenAIAgentConfig(BaseLocalAgentConfig):
  """Configuration for any external OpenAI-compatible completions API (Ollama, LM Studio)."""

  model: str | types.ModelTarget | None = pydantic.Field(
      default=None,
      description="Model identifier or target registered in the local server.",
  )
  base_url: str | None = pydantic.Field(
      default=None,
      description=(
          "Base URL of external server (e.g. http://localhost:11434/v1)."
      ),
  )

  def __init__(
      self,
      *,
      model: str | types.ModelTarget | None = None,
      base_url: str | None = None,
      system_instructions: str | types.SystemInstructions | None = None,
      capabilities: types.CapabilitiesConfig | None = None,
      tools: list[Callable[..., Any]] | None = None,
      policies: list[policy.Policy] | None = None,
      hooks: list[hooks_mod.Hook] | None = None,
      triggers: list[triggers_mod.Trigger] | None = None,
      mcp_servers: list[types.McpServerConfig] | None = None,
      subagents: list[types.SubagentConfig] | None = None,
      workspaces: list[str] | None = None,
      conversation_id: str | None = None,
      save_dir: str | None = None,
      app_data_dir: str | None = None,
      response_schema: (
          dict[str, Any] | type[pydantic.BaseModel] | str | None
      ) = None,
      skills_paths: list[str] | None = None,
      compaction_config: types.CompactionConfig | None = None,
      **kwargs: Any,
  ):
    if capabilities is None:
      capabilities = types.CapabilitiesConfig(
          file_reads=True,
          file_writes=True,
          command_execution=True,
          subagents=True,
          mcp=True,
      )

    init_data = {
        k: v for k, v in locals().items() if k != "self" and v is not None
    }
    if "kwargs" in init_data:
      kwargs_dict = init_data.pop("kwargs")
      if isinstance(kwargs_dict, dict):
        init_data.update(kwargs_dict)
    pydantic.BaseModel.__init__(self, **init_data)

  def create_strategy(
      self,
      *,
      tool_runner: Any,
      hook_runner: Any,
  ) -> "connection.ConnectionStrategy":
    model_name = ""
    resolved_base_url = self.base_url
    if isinstance(self.model, types.ModelTarget):
      model_name = self.model.name or ""
      if not resolved_base_url and self.model.endpoint:
        resolved_base_url = self.model.endpoint.base_url
    elif isinstance(self.model, str):
      model_name = self.model
    # pylint: disable=g-import-not-at-top
    from google.antigravity.connections.local import local_openai_connection

    # pylint: enable=g-import-not-at-top
    return local_openai_connection.LocalOpenAIConnectionStrategy(
        base_url=resolved_base_url or "",
        model_name=model_name,
        tool_runner=tool_runner,
        hook_runner=hook_runner,
        system_instructions=self._get_system_instructions(),
        capabilities_config=self.capabilities,
        compaction_config=self._get_effective_compaction_config(),
        conversation_id=self.conversation_id,
        save_dir=self._get_or_create_save_dir(),
        workspaces=self.workspaces,
        app_data_dir=self.app_data_dir,
        skills_paths=self.skills_paths,
        mcp_servers=self.mcp_servers,
        subagents=self.subagents,
        env=self.env,
        debug_config=self.debug_config,
        retry_config=self.retry_config,
    )
