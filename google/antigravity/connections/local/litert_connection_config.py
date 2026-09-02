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

"""LiteRT Agent Config."""

import enum
from typing import Any, Callable

import pydantic

from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.connections.local.local_connection_config import BaseLocalAgentConfig
from google.antigravity.hooks import hooks as hooks_mod
from google.antigravity.hooks import policy
from google.antigravity.triggers import triggers as triggers_mod


class LiteRTBackend(str, enum.Enum):
  CPU = "cpu"
  GPU = "gpu"
  NPU = "npu"


class LiteRTAgentConfig(BaseLocalAgentConfig):
  """Configuration for local Gemma models using managed LiteRT-LM PyPI backend."""

  model_path: str = pydantic.Field(
      ..., description="Path to the .litertlm model file."
  )
  backend: LiteRTBackend = pydantic.Field(
      default=LiteRTBackend.GPU,
      description="Hardware backend (cpu, gpu, npu).",
  )
  enable_speculative_decoding: bool = pydantic.Field(
      default=False,
      description="Enable speculative decoding (Multi-Token Prediction).",
  )
  cache_dir: str | None = pydantic.Field(
      default=None,
      description="Path to a writable directory for compilation caching.",
  )
  audio_backend: LiteRTBackend | None = pydantic.Field(
      default=None,
      description="Backend override for audio processing.",
  )
  vision_backend: LiteRTBackend | None = pydantic.Field(
      default=None,
      description="Backend override for vision/image processing.",
  )
  port: int = pydantic.Field(
      default=0,
      description=(
          "The port to bind the local server to. 0 picks a random port."
      ),
  )
  download_if_missing: bool = pydantic.Field(
      default=False,
      description="Automatically download weights. Defaults to False.",
  )

  def __init__(
      self,
      *,
      model_path: str,
      backend: LiteRTBackend | str = LiteRTBackend.GPU,
      enable_speculative_decoding: bool = False,
      cache_dir: str | None = None,
      audio_backend: LiteRTBackend | str | None = None,
      vision_backend: LiteRTBackend | str | None = None,
      port: int = 0,
      download_if_missing: bool = False,
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
    if isinstance(backend, str):
      backend = LiteRTBackend(backend.lower())
    if isinstance(audio_backend, str):
      audio_backend = LiteRTBackend(audio_backend.lower())
    if isinstance(vision_backend, str):
      vision_backend = LiteRTBackend(vision_backend.lower())

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
    # pylint: disable=g-import-not-at-top
    try:
      from google.antigravity.connections.local import litert_connection  # type: ignore[missing-module-attribute]
    except (ImportError, ModuleNotFoundError) as e:
      raise RuntimeError(
          "LiteRT backend is not available. To run local LiteRT/Gemma models, "
          "please install the optional LiteRT dependencies (e.g., pip install "
          "litert-lm or include the LiteRT backend in your build dependencies)."
      ) from e

    # pylint: enable=g-import-not-at-top
    return litert_connection.LiteRTConnectionStrategy(
        model_path=self.model_path,
        backend=self.backend,
        enable_speculative_decoding=self.enable_speculative_decoding,
        cache_dir=self.cache_dir,
        audio_backend=self.audio_backend,
        vision_backend=self.vision_backend,
        port=self.port,
        download_if_missing=self.download_if_missing,
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
