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

"""Model configuration types for Google Antigravity SDK.

Defines the types used to configure model backends: which models to use,
how to authenticate, and model-specific options like thinking level.
"""

from __future__ import annotations

import abc
import enum
import os
from typing import Any

import pydantic


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_IMAGE_GENERATION_MODEL = "gemini-3.1-flash-lite-image"


# =============================================================================
# Model types
# =============================================================================


class ThinkingLevel(str, enum.Enum):
  """Thinking level for Gemini models that support extended thinking.

  Controls the amount of reasoning the model performs before responding.
  See https://ai.google.dev/gemini-api/docs/thinking#thinking-levels for
  details.

  Attributes:
    MINIMAL: Minimal thinking.
    LOW: Low thinking.
    MEDIUM: Medium thinking.
    HIGH: High thinking.
    EXTRA_HIGH: Extra-high thinking.
  """

  MINIMAL = "minimal"
  LOW = "low"
  MEDIUM = "medium"
  HIGH = "high"
  EXTRA_HIGH = "extra_high"


class ServiceTier(str, enum.Enum):
  """Service tier for Gemini model inference.

  Controls the compute queue priority and rate limit fallback behavior. See:
  https://ai.google.dev/gemini-api/docs/priority-inference

  Attributes:
    STANDARD: Standard processing tier.
    PRIORITY: Prioritized inference tier.
    FLEX: Flex processing tier.
  """

  STANDARD = "standard"
  PRIORITY = "priority"
  FLEX = "flex"


class ModelType(str, enum.Enum):
  """Discriminator for model purpose."""

  TEXT = "text"
  IMAGE = "image"


class ModelEndpoint(abc.ABC, pydantic.BaseModel):
  """Base class for model endpoint authentication & routing."""

  base_url: str | None = None
  http_headers: dict[str, str] | None = None

  @abc.abstractmethod
  def validate_endpoint(self) -> None:
    """Validates the configuration of the endpoint."""
    pass


class GeminiModelOptions(pydantic.BaseModel):
  """Gemini-specific model options."""

  thinking_level: ThinkingLevel | None = None
  service_tier: ServiceTier | None = None


class GeminiAPIEndpoint(ModelEndpoint):
  """Endpoint for the Gemini Developer API."""

  api_key: str | None = None
  options: GeminiModelOptions | None = None

  def validate_endpoint(self) -> None:
    if self.base_url:
      return  # External API, validation is done by the external API.

    if not (self.api_key or os.environ.get("GEMINI_API_KEY")):
      raise ValueError(
          "A Gemini API key is required. Set it via"
          " GEMINI_API_KEY environment variable or via"
          " LocalAgentConfig(api_key=...)."
      )


class VertexEndpoint(ModelEndpoint):
  """Endpoint for the Vertex AI backend."""

  project: str | None = None
  location: str | None = None
  api_key: str | None = None
  options: GeminiModelOptions | None = None

  @pydantic.model_validator(mode="before")
  @classmethod
  def _populate_env_vars(cls, data: Any) -> Any:
    # Skip populating ambient project/location from the environment when a
    # custom base_url is specified. This matches the Google Gen AI SDK behavior
    # to avoid contaminating external proxies/gateways with ambient shell
    # project metadata (which alters URL paths to /projects/.../locations/...).
    if isinstance(data, dict) and not data.get("base_url"):
      if not data.get("api_key"):
        if data.get("project") is None:
          data["project"] = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if data.get("location") is None:
          data["location"] = os.environ.get("GOOGLE_CLOUD_LOCATION")
    return data

  def validate_endpoint(self) -> None:
    if self.base_url:
      return  # External API, validation is done by the external API.

    has_regional_auth = bool(self.project and self.location)
    has_any_regional_arg = bool(self.project or self.location)
    has_express_auth = bool(self.api_key)

    if has_any_regional_arg and has_express_auth:
      raise ValueError(
          "Cannot specify both api_key (Express Mode) and project/location"
          " (Standard Mode) on VertexEndpoint."
      )

    if not (has_regional_auth or has_express_auth):
      raise ValueError(
          "For Vertex AI, either (project and location) or api_key must be set."
      )


class ModelTarget(pydantic.BaseModel):
  """Configuration for a single model."""

  name: str | None = None
  types: list[ModelType] = pydantic.Field(
      default_factory=lambda: [ModelType.TEXT]
  )
  endpoint: ModelEndpoint | None = None
