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

"""Type definitions for Google Antigravity SDK.

These are the canonical SDK boundary types. All public SDK interfaces use these
types. They are pure Python Pydantic V2 models with no proto dependencies.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import enum
import logging
import mimetypes
import pathlib
from typing import Annotated, Any, AsyncIterator, Callable, ClassVar, Literal, TypeVar, cast
import warnings

import pydantic

from google.antigravity.models import GeminiAPIEndpoint
from google.antigravity.models import GeminiModelOptions
from google.antigravity.models import ModelEndpoint
from google.antigravity.models import ModelTarget
from google.antigravity.models import ModelType
from google.antigravity.models import ServiceTier
from google.antigravity.models import ThinkingLevel
from google.antigravity.models import VertexEndpoint

_BaseMediaT = TypeVar("_BaseMediaT", bound="_BaseMedia")

__all__ = [
    "ThinkingLevel",
    "ServiceTier",
    "ModelType",
    "ModelEndpoint",
    "GeminiAPIEndpoint",
    "VertexEndpoint",
    "GeminiModelOptions",
    "ModelTarget",
    "SystemInstructionSection",
    "CustomSystemInstructions",
    "TemplatedSystemInstructions",
    "SystemInstructions",
    "SubagentConfig",
    "SubagentCapabilities",
    "AgentBehavior",
    "BuiltinTools",
    "RunCommandConfig",
    "CapabilitiesConfig",
    "ModelAPIRetryConfig",
    "ModelOutputRetryConfig",
    "RetryConfig",
    "SessionContinuationMode",
    "BaseMcpServerConfig",
    "McpStdioServer",
    "McpStreamableHttpServer",
    "McpServerConfig",
    "ToolCall",
    "ToolResult",
    "UsageMetadata",
    "StepType",
    "StepSource",
    "StepTarget",
    "StepStatus",
    "BudgetScope",
    "BudgetConfig",
    "StopReason",
    "Step",
    "HookResult",
    "QuestionResponse",
    "QuestionHookResult",
    "AskQuestionOption",
    "AskQuestionEntry",
    "AskQuestionInteractionSpec",
    "AntigravityConnectionError",
    "AntigravityCancelledError",
    "AntigravityValidationError",
    "AntigravityExecutionError",
    "ToolExecutionError",
    "StreamChunk",
    "Thought",
    "Text",
    "ChatResponse",
    "Image",
    "Document",
    "Audio",
    "Video",
    "Content",
    "ContentPrimitive",
    "from_file",
    "from_bytes",
    "SlashCommand",
    "BuiltinSlashCommandName",
    "StopDecision",
    "StopHookResult",
    "StopArgs",
]

# =============================================================================
# Config types
# =============================================================================


class SystemInstructionSection(pydantic.BaseModel):
  """A named section to append to the system instructions."""

  content: str
  title: str = "user_system_instructions"


class CustomSystemInstructions(pydantic.BaseModel):
  """Use this to completely replace the system instructions.

  WARNING: For advanced usage only. This replaces ALL default instructions.
  If you use this, you are responsible for providing all necessary instructions
  yourself, for example:
  - **Core Mandates**: Security and safety rules (e.g., credential protection).
  - **Engineering Standards**: Coding style, testing, and linting rules.
  - **Operational Guidelines**: Tone, brevity, and tool usage protocols.

  Most users should use TemplatedSystemInstructions instead.
  """

  text: str


class TemplatedSystemInstructions(pydantic.BaseModel):
  """Use this to override the agent's identity and append sections to the default system instructions.

  See `examples/getting_started/persona_config.py`
  for a full example with identity and sections.
  """

  identity: str | None = None
  sections: list[SystemInstructionSection] = pydantic.Field(
      default_factory=list
  )


# Union type representing the two ways to configure system instructions.
# - CustomSystemInstructions: Full replacement (Advanced usage).
# - TemplatedSystemInstructions: Append to defaults (Recommended).
SystemInstructions = CustomSystemInstructions | TemplatedSystemInstructions


class AgentBehavior(str, enum.Enum):
  """Operational execution behavior for an agent.

  Attributes:
    AUTONOMOUS: Non-interactive, automated execution. The agent must accomplish
      the task on its own.
    INTERACTIVE: The agent works collaboratively with a human, asking for
      clarifications and keeping them in the loop if needed. Enables features
      like slash commands and planning mode.
    MINIMAL: Streamlined prompt behavior optimized for small-context and
      on-device models by filtering system instructions down to core identity,
      guidelines, and communication style.
  """

  AUTONOMOUS = "autonomous"
  INTERACTIVE = "interactive"
  MINIMAL = "minimal"


class RunCommandConfig(pydantic.BaseModel):
  """Configuration for the builtin run_command tool.

  Attributes:
    enable_daemons: Whether the agent is authorized to start long-running daemon
      commands (e.g. background dev servers, watchers) using
      run_command(IsDaemon=True) without blocking session completion. When True,
      the IsDaemon argument is exposed on the run_command tool schema. Defaults
      to False.
    timeout_seconds: Maximum execution duration in seconds for commands. When
      None, the default timeout (10 minutes) is used. Defaults to None.
    enable_sandbox: When True, terminal commands (run_command) are executed
      inside the OS-level sandbox (exebox). Forwarded to the harness/cortex,
      which enforces the sandbox at command execution time. Has no effect on
      platforms/environments where the sandbox is unavailable. Defaults to
      False.
  """

  enable_daemons: bool = False
  timeout_seconds: float | None = pydantic.Field(default=None, gt=0)
  enable_sandbox: bool = False


class SubagentCapabilities(pydantic.BaseModel):
  """Capabilities configuration for subagents.

  Attributes:
    agent_behavior: Operational execution behavior for the subagent. In
      particular, AgentBehavior.AUTONOMOUS incentivizes the agent to solve the
      task on their own from start to finish, AgentBehavior.INTERACTIVE makes
      the agent work collaboratively with a human, and AgentBehavior.MINIMAL
      prunes prompt overhead for small-context models. Defaults to
      AgentBehavior.AUTONOMOUS.
    allowed_subagents: Explicit allowlist of subagent names this subagent may
      directly invoke. When None, all registered subagents are discoverable.
    enabled_tools: Explicit allowlist of builtin tools to enable. Mutually
      exclusive with disabled_tools. When None, the harness defaults are used.
    disabled_tools: Explicit denylist of builtin tools to disable. Mutually
      exclusive with enabled_tools. When None, the harness defaults are used.
    run_command_config: Optional configuration for the builtin run_command tool.
  """

  agent_behavior: AgentBehavior = AgentBehavior.AUTONOMOUS
  allowed_subagents: list[str] | None = None
  enabled_tools: list[BuiltinTools] | None = None
  disabled_tools: list[BuiltinTools] | None = None
  run_command_config: RunCommandConfig | None = None

  @pydantic.model_validator(mode="after")
  def _check_mutually_exclusive(self) -> "SubagentCapabilities":
    if self.enabled_tools is not None and self.disabled_tools is not None:
      raise ValueError(
          "enabled_tools and disabled_tools should be mutually exclusive."
      )
    return self

  @pydantic.model_validator(mode="after")
  def _validate_subagents_and_tools(self) -> "SubagentCapabilities":
    subagent_disabled = (
        self.disabled_tools is not None
        and BuiltinTools.START_SUBAGENT in self.disabled_tools
    ) or (
        self.enabled_tools is not None
        and BuiltinTools.START_SUBAGENT not in self.enabled_tools
    )
    if subagent_disabled and self.allowed_subagents is not None:
      raise ValueError(
          "allowed_subagents cannot be specified when START_SUBAGENT is"
          " disabled or omitted from enabled_tools."
      )
    return self

  @pydantic.model_validator(mode="after")
  def _validate_interactive_tools(self) -> "SubagentCapabilities":
    if (
        self.enabled_tools is not None
        and BuiltinTools.ASK_QUESTION in self.enabled_tools
        and self.agent_behavior != AgentBehavior.INTERACTIVE
    ):
      logging.warning(
          "BuiltinTools.ASK_QUESTION is enabled on subagent, but"
          " agent_behavior is not INTERACTIVE. Set"
          " SubagentCapabilities(agent_behavior=AgentBehavior.INTERACTIVE) if"
          " interactive question-and-answer behavior is desired."
      )
    return self


class SubagentConfig(pydantic.BaseModel):
  """Configuration for a static subagent.

  Attributes:
    name: Unique name of the subagent.
    description: Description of the subagent.
    system_instructions: Optional system instructions for the subagent. Supports
      a string or a SystemInstructions. Note that string and
      TemplatedSystemInstructions inputs will be appended to the subagent's
      default system instructions, whereas CustomSystemInstructions will
      completely replace them.
    capabilities: Optional capabilities config controlling allowed tools. If
      None, defaults to read-only tools.
    tools: Optional list of additional custom tools (callable functions or
      string names) to enable for this subagent.
  """

  name: str
  description: str
  system_instructions: str | SystemInstructions | None = None
  capabilities: SubagentCapabilities | None = None
  tools: list[Callable[..., Any] | str] | None = pydantic.Field(
      default_factory=list
  )


class BuiltinTools(str, enum.Enum):
  """Identifiers for common connection-provided builtin tools.

  Attributes:
    LIST_DIR: List directory contents.
    SEARCH_DIR: Search within directories (grep).
    FIND_FILE: Find files by name within a directory.
    VIEW_FILE: View file contents.
    CREATE_FILE: Create a new file.
    EDIT_FILE: Edit an existing file.
    RUN_COMMAND: Execute a shell command.
    ASK_QUESTION: Ask the user a clarifying question.
    START_SUBAGENT: Invoke a subagent.
    GENERATE_IMAGE: Generate or edit images.
    SEARCH_WEB: Search the web.
    READ_URL_CONTENT: Read content from a URL.
    FINISH: Finish the conversation and return structured output.
  """

  LIST_DIR = "list_directory"
  SEARCH_DIR = "search_directory"
  FIND_FILE = "find_file"
  VIEW_FILE = "view_file"
  CREATE_FILE = "create_file"
  EDIT_FILE = "edit_file"
  RUN_COMMAND = "run_command"
  ASK_QUESTION = "ask_question"
  START_SUBAGENT = "start_subagent"
  GENERATE_IMAGE = "generate_image"
  SEARCH_WEB = "search_web"
  READ_URL_CONTENT = "read_url_content"
  FINISH = "finish"

  @classmethod
  def read_only(cls) -> list["BuiltinTools"]:
    """Returns tools that only read state (no writes, deletes, or commands).

    Returns:
        A list of read-only BuiltinTools.
    """
    return [
        cls.LIST_DIR,
        cls.SEARCH_DIR,
        cls.FIND_FILE,
        cls.VIEW_FILE,
        cls.READ_URL_CONTENT,
        cls.FINISH,
    ]

  @classmethod
  def nondestructive(cls) -> list["BuiltinTools"]:
    """Returns tools that cannot delete content.

    Returns:
        A list of non-destructive BuiltinTools.
    """
    return [
        cls.LIST_DIR,
        cls.SEARCH_DIR,
        cls.FIND_FILE,
        cls.VIEW_FILE,
        cls.CREATE_FILE,
        cls.EDIT_FILE,
        cls.ASK_QUESTION,
        cls.START_SUBAGENT,
        cls.GENERATE_IMAGE,
        cls.SEARCH_WEB,
        cls.READ_URL_CONTENT,
        cls.FINISH,
    ]

  @classmethod
  def all_tools(cls) -> list["BuiltinTools"]:
    """Returns all builtin tools.

    Returns:
        A list of all BuiltinTools.
    """
    return list(cls)

  @classmethod
  def file_tools(cls) -> list["BuiltinTools"]:
    """Returns tools that perform file read/write/create operations.

    These tools accept a file path argument and can be scoped to specific
    workspace directories via ``policy.workspace_only()``.

    Returns:
        A list of file-operation BuiltinTools.
    """
    return [
        cls.VIEW_FILE,
        cls.CREATE_FILE,
        cls.EDIT_FILE,
    ]

  @classmethod
  def none(cls) -> list["BuiltinTools"]:
    """Returns an empty tool list (no builtin tools).

    Returns:
        An empty list of BuiltinTools.
    """
    return []

  @classmethod
  def minimal(cls) -> list["BuiltinTools"]:
    """Returns the minimal set of software engineering tools.

    Includes run_command, view_file, create_file, edit_file, list_directory, and
    search_directory.

    Returns:
        A list of minimal BuiltinTools.
    """
    return [
        cls.RUN_COMMAND,
        cls.VIEW_FILE,
        cls.CREATE_FILE,
        cls.EDIT_FILE,
        cls.LIST_DIR,
        cls.SEARCH_DIR,
    ]


class CapabilitiesConfig(pydantic.BaseModel):
  """General agent capability configuration.

  Disabling vs. Denying Tools:

    ``enabled_tools`` / ``disabled_tools`` control which tools the harness
    *exposes* to the model. A disabled tool is stripped from the model's
    context entirely — the model never sees it, never wastes tokens
    considering it, and never attempts to call it. Use these fields when
    a tool is irrelevant to the agent's purpose.

    By contrast, the policy system (``hooks.policy.deny()``) leaves a tool
    visible in the model's context but rejects the call at runtime. The
    model may still attempt to invoke a policy-denied tool, at which point
    the SDK returns a denial message. This costs tokens and may cause
    retries, but it allows the model to understand *why* access was
    refused, which can be useful for adaptive agents.

    **Guideline**: Prefer ``disabled_tools`` / ``enabled_tools`` for tools
    the agent should never use. Use ``policy.deny()`` for conditional or
    context-dependent restrictions (e.g., blocking ``run_command`` only
    when the arguments match a dangerous pattern).

  Attributes:
    enable_subagents: Whether the agent can spawn and delegate to sub-agents.
    agent_behavior: Operational execution behavior for the agent. In particular,
      AgentBehavior.AUTONOMOUS incentivizes the agent to solve the task on their
      own from start to finish, AgentBehavior.INTERACTIVE makes the agent work
      collaboratively with a human, and AgentBehavior.MINIMAL prunes prompt
      overhead for small-context models. Defaults to AgentBehavior.AUTONOMOUS.
    enabled_tools: Explicit allowlist of builtin tools to enable. Mutually
      exclusive with disabled_tools. When None, the harness defaults are used
      (all tools enabled). Disabled tools are removed from the model's context,
      saving tokens and preventing the model from even considering them.
    disabled_tools: Explicit denylist of builtin tools to disable. Mutually
      exclusive with enabled_tools. When None, the harness defaults are used
      (all tools enabled). Disabled tools are removed from the model's context,
      saving tokens and preventing the model from even considering them.
    compaction_threshold: (Deprecated) Configure
      CompactionConfig(checkpoint_interval_tokens=...) directly on AgentConfig
      instead.
    finish_tool_schema_json: Optional JSON schema string for the finish tool.
    max_subagent_depth: Global maximum subagent recursion depth for the session.
      When None, defaults to 1 (flat single-level delegation).
    allowed_subagents: Explicit allowlist of subagent names the root agent may
      directly invoke. When None, all registered subagents are discoverable.
    run_command_config: Optional configuration for the builtin run_command tool.
  """

  enable_subagents: bool = True
  agent_behavior: AgentBehavior = AgentBehavior.AUTONOMOUS
  enabled_tools: list[BuiltinTools] | None = None
  disabled_tools: list[BuiltinTools] | None = None
  compaction_threshold: int | None = pydantic.Field(
      default=None,
      gt=0,
      deprecated=(
          "CapabilitiesConfig.compaction_threshold is deprecated. Configure"
          " CompactionConfig(checkpoint_interval_tokens=...) directly on"
          " AgentConfig instead."
      ),
  )
  finish_tool_schema_json: str | None = None
  max_subagent_depth: int | None = pydantic.Field(default=None, ge=1)
  allowed_subagents: list[str] | None = None
  run_command_config: RunCommandConfig | None = None

  @pydantic.model_validator(mode="after")
  def _check_mutually_exclusive(self) -> "CapabilitiesConfig":
    if self.enabled_tools is not None and self.disabled_tools is not None:
      raise ValueError(
          "enabled_tools and disabled_tools should be mutually exclusive."
      )
    return self

  @pydantic.model_validator(mode="after")
  def _validate_subagents_and_tools(self) -> "CapabilitiesConfig":
    subagent_disabled = (
        not self.enable_subagents
        or (
            self.disabled_tools is not None
            and BuiltinTools.START_SUBAGENT in self.disabled_tools
        )
        or (
            self.enabled_tools is not None
            and BuiltinTools.START_SUBAGENT not in self.enabled_tools
        )
    )
    if subagent_disabled:
      if self.max_subagent_depth is not None:
        raise ValueError(
            "max_subagent_depth cannot be configured when subagents are"
            " disabled (enable_subagents=False or START_SUBAGENT not enabled)."
        )
      if self.allowed_subagents is not None:
        raise ValueError(
            "allowed_subagents cannot be specified when subagents are disabled."
        )
    return self

  @pydantic.model_validator(mode="after")
  def _validate_interactive_tools(self) -> "CapabilitiesConfig":
    if (
        self.enabled_tools is not None
        and BuiltinTools.ASK_QUESTION in self.enabled_tools
        and self.agent_behavior != AgentBehavior.INTERACTIVE
    ):
      logging.warning(
          "BuiltinTools.ASK_QUESTION is enabled, but agent_behavior is not"
          " INTERACTIVE. Set"
          " CapabilitiesConfig(agent_behavior=AgentBehavior.INTERACTIVE) if"
          " interactive question-and-answer behavior is desired."
      )
    return self

  @pydantic.model_validator(mode="after")
  def _warn_deprecated_compaction_fields(self) -> "CapabilitiesConfig":
    if self.compaction_threshold is not None:
      warnings.warn(
          "CapabilitiesConfig.compaction_threshold is deprecated. Configure"
          " CompactionConfig(checkpoint_interval_tokens=...) directly on"
          " AgentConfig instead.",
          category=DeprecationWarning,
          stacklevel=2,
      )
    return self


class CompactionConfig(pydantic.BaseModel):
  """Configuration for conversation trajectory compaction and context limits.

  Antigravity manages context using a two-stage sliding-window pipeline:
  1. Background Checkpointing: A "checkpoint" is an asynchronous summary of the
     conversation trajectory prepared in the background while the agent works.
     Every checkpoint is cumulative, summarizing history up to that point.
     Generating a checkpoint does not modify the active prompt or evict turns.
  2. Prompt Eviction (Compaction): When active history reaches the token
     ceiling (`max_context_tokens`), the context snaps back to the latest
     completed checkpoint. Earlier checkpoints and turns preceding the latest
     checkpoint are evicted from the prompt, while recent turns between the
     latest checkpoint and the current turn are preserved verbatim with full
     fidelity.

  Attributes:
    checkpoint_interval_tokens: The token interval at which background
      trajectory checkpoints (summaries) are pre-computed. Checkpoint generation
      runs asynchronously and does not alter the active prompt. When None, the
      framework's default cadence is used.
    max_context_tokens: Maximum token ceiling allowed for the prompt before
      older turns are evicted and replaced with the latest checkpoint summary.
      When None, the framework's default limit is used.
    compaction_threshold: Deprecated alias for `checkpoint_interval_tokens`.
  """

  checkpoint_interval_tokens: int | None = pydantic.Field(default=None, gt=0)
  max_context_tokens: int | None = pydantic.Field(default=None, gt=0)
  compaction_threshold: int | None = pydantic.Field(
      default=None,
      gt=0,
      deprecated=(
          "CompactionConfig.compaction_threshold is deprecated. Use"
          " checkpoint_interval_tokens instead."
      ),
  )

  @pydantic.model_validator(mode="after")
  def _validate_compaction_and_context_tokens(self) -> "CompactionConfig":
    if (
        self.checkpoint_interval_tokens is None
        and self.compaction_threshold is not None
    ):
      self.__dict__["checkpoint_interval_tokens"] = self.compaction_threshold
      self.__pydantic_fields_set__.add("checkpoint_interval_tokens")
    elif (
        self.checkpoint_interval_tokens is not None
        and self.compaction_threshold is None
    ):
      self.__dict__["compaction_threshold"] = self.checkpoint_interval_tokens
      self.__pydantic_fields_set__.add("compaction_threshold")
    elif (
        self.checkpoint_interval_tokens is not None
        and self.compaction_threshold is not None
        and self.checkpoint_interval_tokens != self.compaction_threshold
    ):
      raise ValueError(
          "Conflicting values for aliased fields:"
          f" checkpoint_interval_tokens={self.checkpoint_interval_tokens}"
          f" vs compaction_threshold={self.compaction_threshold}"
      )
    interval = self.checkpoint_interval_tokens
    if (
        interval is not None
        and self.max_context_tokens is not None
        and interval > self.max_context_tokens
    ):
      raise ValueError(
          f"checkpoint_interval_tokens ({interval}) cannot exceed"
          f" max_context_tokens ({self.max_context_tokens})"
      )
    return self


_MAX_INT32 = 2**31 - 1  # Maximum value for protobuf int32 wire fields
_MAX_UINT32 = 2**32 - 1  # Maximum value for protobuf uint32 wire fields
_MAX_INT64 = 2**63 - 1  # Maximum value for protobuf int64 wire fields


class ModelAPIRetryConfig(pydantic.BaseModel):
  """Configuration for API retry behavior with exponential backoff.

  Attributes:
    max_retries: The maximum number of retries for transient API errors.
    initial_sleep_duration_ms: The initial sleep duration in milliseconds.
    exponential_multiplier: The multiplier for exponential backoff.
    jitter_range: The range for jitter when calculating backoff.
  """

  max_retries: int | None = pydantic.Field(default=None, ge=0, le=_MAX_UINT32)
  initial_sleep_duration_ms: int | None = pydantic.Field(
      default=None, ge=0, le=_MAX_UINT32
  )
  exponential_multiplier: float | None = pydantic.Field(default=None, ge=0.0)
  jitter_range: float | None = pydantic.Field(default=None, ge=0.0)


class ModelOutputRetryConfig(pydantic.BaseModel):
  """Configuration for model output retry behavior.

  Attributes:
    max_retries: The maximum number of retries for malformed model outputs.
  """

  max_retries: int | None = pydantic.Field(default=None, ge=0, le=_MAX_UINT32)


class RetryConfig(pydantic.BaseModel):
  """Combined retry configuration for model API calls and output validation.

  When `retry_config` is omitted (or fields are left as None), the backend
  automatically applies built-in interactive defaults (e.g., standard API retry
  counts, exponential backoff, and model output validation attempts). You only
  need to provide explicit configuration when overriding these system defaults.

  Attributes:
    api_retry: Optional configuration for API retry behavior with exponential
      backoff.
    model_output_retry: Optional configuration for model output retry behavior.
  """

  api_retry: ModelAPIRetryConfig | None = None
  model_output_retry: ModelOutputRetryConfig | None = None

  @classmethod
  def benchmark(cls) -> "RetryConfig":
    """Optimized for evaluation suites, automated benchmarks, and load testing.

    Uses unbounded retry tolerance (max uint32: 4,294,967,295 attempts) for
    transient API errors (429 rate limits, 503 service throttling) to prevent
    quota issues from crashing evaluation suites, while relying on the default
    localharness model output retry behavior to remain consistent with
    production product performance.

    Returns:
      A RetryConfig configured for benchmark and eval workflows.
    """
    return cls(
        api_retry=ModelAPIRetryConfig(
            max_retries=_MAX_UINT32, initial_sleep_duration_ms=1000
        )
    )


class BaseMcpServerConfig(pydantic.BaseModel):
  """Base configuration for all Model Context Protocol (MCP) servers.

  Attributes:
    name: Unique identifier for the MCP server. Must match the regex pattern
      ^[a-zA-Z0-9_-]+$, which aligns with the naming constraints of the Gemini
      API tool naming specification (only alphanumeric characters, hyphens, and
      underscores are permitted).
    timeout_seconds: Optional timeout in seconds for connecting to the server
      and listing tools.
    enabled_tools: Explicit allowlist of tools to enable. Mutually exclusive
      with disabled_tools. When None, all tools from the server are enabled.
      Only enabled tools are exposed to the model; others are hidden entirely
      from the model's context, saving tokens.
    disabled_tools: Explicit denylist of tools to disable. Mutually exclusive
      with enabled_tools. When None, all tools from the server are enabled.
      Disabled tools are removed from the model's context entirely, saving
      tokens and preventing the model from even considering them.
  """

  name: Annotated[str, pydantic.Field(pattern=r"^[a-zA-Z0-9_-]+$")]
  timeout_seconds: int | None = None
  enabled_tools: list[str] | None = None
  disabled_tools: list[str] | None = None

  @pydantic.model_validator(mode="after")
  def _check_mutually_exclusive(self) -> "BaseMcpServerConfig":
    if self.enabled_tools is not None and self.disabled_tools is not None:
      raise ValueError(
          "enabled_tools and disabled_tools should be mutually exclusive."
      )
    return self


class McpStdioServer(BaseMcpServerConfig):
  """Configuration for an MCP server connected via stdio.

  Attributes:
    command: The command to run to start the server.
    name: Unique identifier for this MCP server.
    type: The type of connection, always "stdio".
    args: Arguments to pass to the command.
    env: Environment variables to merge into the spawned subprocess's
      environment.
  """

  command: str
  type: Literal["stdio"] = "stdio"
  args: list[str] = pydantic.Field(default_factory=list)
  env: dict[str, str] | None = None


class McpStreamableHttpServer(BaseMcpServerConfig):
  """Configuration for an MCP server connected via Streamable HTTP.

  Attributes:
    url: The URL of the HTTP endpoint.
    name: Unique identifier for this MCP server.
    type: The type of connection, always "http".
    headers: Optional headers to send with the connection request.
    timeout: Connection timeout in seconds.
    sse_read_timeout: SSE read timeout in seconds.
    terminate_on_close: Whether to terminate the connection on close.
  """

  url: str
  type: Literal["http"] = "http"
  headers: dict[str, str] | None = None
  timeout: float = 30.0
  sse_read_timeout: float = 300.0
  terminate_on_close: bool = True


McpServerConfig = McpStdioServer | McpStreamableHttpServer


# =============================================================================
# Tool types
# =============================================================================


class ToolCall(pydantic.BaseModel):
  """A tool call to inject into the conversation.

  Attributes:
    id: Optional unique identifier for the call, often assigned by the backend.
    step_id: Optional identifier correlating this call with its step in the
      trajectory.
    name: Tool identifier. Use a BuiltinTools member for Connection-provided
      tools, or an arbitrary string for custom host-side tools.
    args: Keyword arguments for the tool, as a JSON-serializable dict.
    canonical_path: Optional normalized filesystem path for file-related tools.
      Populated by the Connection layer to enable platform-agnostic L2 policies.
    server_name: Optional server name if this tool belongs to an MCP server.
  """

  name: BuiltinTools | str
  args: dict[str, Any] = pydantic.Field(default_factory=dict)
  id: str | None = None
  step_id: str | None = None
  canonical_path: str | None = None
  server_name: str | None = None


class ToolResult(pydantic.BaseModel):
  """Result of a single tool execution.

  Attributes:
    id: Optional identifier correlating this result with a ToolCall.id.
    step_id: Optional step identifier correlating this result with a step.
    name: The name of the tool that was executed. A BuiltinTools member for
      Connection-provided tools, or a string for custom host-side tools.
    result: The tool's return value. Can be any JSON-serializable value.
    error: An error message if execution failed, or None on success.
    exception: The original exception if execution failed. Not serialized.
    server_name: Optional server name if this tool belongs to an MCP server.
  """

  model_config = pydantic.ConfigDict(
      extra="ignore", arbitrary_types_allowed=True
  )

  name: BuiltinTools | str
  id: str | None = None
  step_id: str | None = None
  result: Any = None
  error: str | None = None
  exception: Exception | None = pydantic.Field(default=None, exclude=True)
  server_name: str | None = None


PythonTool = Callable[..., Any]


# =============================================================================
# Step types
# =============================================================================


class UsageMetadata(pydantic.BaseModel):
  """Token usage metadata from the model API.

  Fields are None when the data is not available (e.g. the step did not
  involve a model call). A value of 0 means the model explicitly reported
  zero tokens for that category.

  Attributes:
    prompt_token_count: Number of tokens in the prompt.
    cached_content_token_count: Number of tokens from cached content. These are
      a subset of prompt tokens.
    candidates_token_count: Number of tokens in the generated candidates
      (excluding thinking).
    thoughts_token_count: Number of tokens used for thinking/reasoning.
    total_token_count: Sum of prompt + candidates + thinking tokens.
    service_tier: Service tier used for inference (e.g. "priority").
  """

  # Input tokens.
  prompt_token_count: int | None = None
  cached_content_token_count: int | None = None

  # Output tokens.
  candidates_token_count: int | None = None
  thoughts_token_count: int | None = None

  # Total tokens (prompt + candidates + thoughts).
  total_token_count: int | None = None

  # Service tier.
  service_tier: ServiceTier | None = None

  def __add__(self, other: UsageMetadata) -> UsageMetadata:
    if not isinstance(other, UsageMetadata):
      return NotImplemented
    if self.service_tier == other.service_tier:
      merged_tier = self.service_tier
    elif self.service_tier is None or other.service_tier is None:
      merged_tier = self.service_tier or other.service_tier
    else:
      # When combining different service tiers, default to STANDARD.
      merged_tier = ServiceTier.STANDARD
    return UsageMetadata(
        prompt_token_count=(self.prompt_token_count or 0)
        + (other.prompt_token_count or 0),
        cached_content_token_count=(self.cached_content_token_count or 0)
        + (other.cached_content_token_count or 0),
        candidates_token_count=(self.candidates_token_count or 0)
        + (other.candidates_token_count or 0),
        thoughts_token_count=(self.thoughts_token_count or 0)
        + (other.thoughts_token_count or 0),
        total_token_count=(self.total_token_count or 0)
        + (other.total_token_count or 0),
        service_tier=merged_tier,
    )

  def __sub__(self, other: UsageMetadata) -> UsageMetadata:
    if not isinstance(other, UsageMetadata):
      return NotImplemented
    return UsageMetadata(
        prompt_token_count=(self.prompt_token_count or 0)
        - (other.prompt_token_count or 0),
        cached_content_token_count=(self.cached_content_token_count or 0)
        - (other.cached_content_token_count or 0),
        candidates_token_count=(self.candidates_token_count or 0)
        - (other.candidates_token_count or 0),
        thoughts_token_count=(self.thoughts_token_count or 0)
        - (other.thoughts_token_count or 0),
        total_token_count=(self.total_token_count or 0)
        - (other.total_token_count or 0),
        service_tier=self.service_tier or other.service_tier,
    )


class StepType(str, enum.Enum):
  """High-level type of a step."""

  TEXT_RESPONSE = "TEXT_RESPONSE"
  TOOL_CALL = "TOOL_CALL"
  SYSTEM_MESSAGE = "SYSTEM_MESSAGE"
  COMPACTION = "COMPACTION"
  FINISH = "FINISH"
  THINKING = "THINKING"
  UNKNOWN = "UNKNOWN"


class StepSource(str, enum.Enum):
  """Source of a step."""

  SYSTEM = "SYSTEM"
  USER = "USER"
  MODEL = "MODEL"
  UNKNOWN = "UNKNOWN"


class StepTarget(str, enum.Enum):
  """Target of a step interaction."""

  USER = "TARGET_USER"
  ENVIRONMENT = "TARGET_ENVIRONMENT"
  UNSPECIFIED = "TARGET_UNSPECIFIED"
  UNKNOWN = "UNKNOWN"


class StepStatus(str, enum.Enum):
  """Status of a step."""

  ACTIVE = "ACTIVE"
  DONE = "DONE"
  WAITING_FOR_USER = "WAITING_FOR_USER"
  ERROR = "ERROR"
  CANCELED = "CANCELED"
  UNKNOWN = "UNKNOWN"


class SessionContinuationMode(str, enum.Enum):
  """Mode for establishing a connection to an agent session.

  Attributes:
    RESUME: Resume an existing session. Fail if it doesn't exist.
    CREATE_OR_RESUME: Resume if exists, create a new one if missing.
    CREATE_ONLY: Create a new session. Fail if it already exists.
  """

  RESUME = "resume"
  CREATE_OR_RESUME = "create_or_resume"
  CREATE_ONLY = "create_only"


class BudgetScope(str, enum.Enum):
  """Evaluation scope for budget limits and caps.

  Attributes:
    LIFETIME: Budget is evaluated against cumulative spend from the start of the
      session (step 0).
    FORWARD_LOOKING: Budget is evaluated against spend starting from when the
      budget was configured or the session was resumed.
  """

  LIFETIME = "LIFETIME"
  FORWARD_LOOKING = "FORWARD_LOOKING"


class BudgetConfig(pydantic.BaseModel):
  """Configuration for session-level budget limits and caps.

  Attributes:
    max_model_calls: Maximum number of model invocations (reasoning steps /
      generator calls) permitted within the configured scope.
    max_tool_calls: Maximum number of tool invocations permitted within the
      configured scope, regardless of tool source.
    max_input_tokens: Maximum net uncached input tokens permitted within the
      configured scope (calculated as prompt tokens minus cached content tokens
      across model turns).
    max_output_tokens: Maximum output tokens permitted within the configured
      scope (candidates + thoughts).
    max_total_tokens: Maximum total net tokens permitted within the configured
      scope (calculated as net uncached input tokens + output tokens
      across model turns).
    scope: The evaluation scope for this budget configuration. Defaults to
      BudgetScope.LIFETIME.
  """

  max_model_calls: int | None = pydantic.Field(
      default=None, ge=1, le=_MAX_INT32
  )
  max_tool_calls: int | None = pydantic.Field(default=None, ge=1, le=_MAX_INT32)
  max_input_tokens: int | None = pydantic.Field(
      default=None, ge=1, le=_MAX_INT64
  )
  max_output_tokens: int | None = pydantic.Field(
      default=None, ge=1, le=_MAX_INT64
  )
  max_total_tokens: int | None = pydantic.Field(
      default=None, ge=1, le=_MAX_INT64
  )
  scope: BudgetScope = BudgetScope.LIFETIME


class StopReason(str, enum.Enum):
  """Reason why the execution turn stopped.

  Attributes:
    UNSPECIFIED: Default value; normal completion or unspecified stop reason.
    MAX_MODEL_CALLS_EXCEEDED: Turn halted because session exceeded configured
      max_model_calls.
    MAX_TOOL_CALLS_EXCEEDED: Turn halted because session exceeded
      max_tool_calls.
    MAX_INPUT_TOKENS_EXCEEDED: Turn halted because session exceeded
      max_input_tokens.
    MAX_OUTPUT_TOKENS_EXCEEDED: Turn halted because session exceeded
      max_output_tokens.
    MAX_TOTAL_TOKENS_EXCEEDED: Turn halted because session exceeded
      max_total_tokens.
    QUOTA_EXHAUSTED: Turn halted because backend model API quota was exhausted.
  """

  UNSPECIFIED = "UNSPECIFIED"
  MAX_MODEL_CALLS_EXCEEDED = "MAX_MODEL_CALLS_EXCEEDED"
  MAX_TOOL_CALLS_EXCEEDED = "MAX_TOOL_CALLS_EXCEEDED"
  MAX_INPUT_TOKENS_EXCEEDED = "MAX_INPUT_TOKENS_EXCEEDED"
  MAX_OUTPUT_TOKENS_EXCEEDED = "MAX_OUTPUT_TOKENS_EXCEEDED"
  MAX_TOTAL_TOKENS_EXCEEDED = "MAX_TOTAL_TOKENS_EXCEEDED"
  QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"


class Step(pydantic.BaseModel):
  """Structure representing one action in the agent trajectory.

  Attributes:
    id: Unique string identifier for the step.
    step_index: Integer index of the step in the trajectory.
    trajectory_id: Unique identifier of the trajectory owning this step.
    parent_trajectory_id: ID of the parent trajectory that spawned this step, or
      empty string for the root agent conversation.
    depth: Nesting depth of this step (0 for root conversation).
    type: The high-level type of the step.
    source: The source that generated the step.
    target: The target interacting with this step.
    status: The status of the step.
    content: The output of the step.
    thinking: Model reasoning/thinking for planner responses.
    content_delta: Text added since the last update for this step.
    thinking_delta: Thinking added since the last update for this step.
    tool_calls: List of tool calls associated with the step.
    error: Short error message if the step failed or empty string.
    is_complete_response: True if this step is a completed model response
      directed at the user, as distinct from a partial streaming chunk. Multiple
      steps per turn may have this flag set; consumers that want only the last
      response should iterate fully.
    structured_output: The structured output extracted from the finish step.
    usage_metadata: (Deprecated) Token usage for this specific step's model
      invocation. Deprecated in favor of ChatResponse.usage_metadata (turn-level
      usage) and agent.conversation.total_usage (session cumulative usage).
  """

  id: str = ""
  step_index: int = 0
  trajectory_id: str = ""
  parent_trajectory_id: str = ""
  depth: int = 0
  type: StepType = StepType.UNKNOWN
  source: StepSource = StepSource.UNKNOWN
  target: StepTarget = StepTarget.UNKNOWN
  status: StepStatus = StepStatus.UNKNOWN
  content: str = ""
  content_delta: str = ""
  thinking: str = ""
  thinking_delta: str = ""
  tool_calls: list[ToolCall] = pydantic.Field(default_factory=list)
  error: str = ""
  is_complete_response: bool | None = None
  structured_output: Any | None = None
  usage_metadata: UsageMetadata | None = pydantic.Field(
      default=None,
      deprecated=(
          "Step.usage_metadata is deprecated and will be removed in a future"
          " release. Token usage is emitted per model invocation and does not"
          " map 1:1 to individual execution steps. Use"
          " ChatResponse.usage_metadata for turn-level usage or"
          " agent.conversation.total_usage for cumulative session usage."
      ),
  )

  model_config = pydantic.ConfigDict(extra="allow")


# =============================================================================
# Hook types
# =============================================================================
class HookResult(pydantic.BaseModel):
  """Result of a decision hook execution.

  Attributes:
    allow: Whether execution should proceed.
    message: Optional explanation or response message.
    modified_args: Optional dictionary of modified tool arguments to
      shallow-merge into the existing arguments dictionary (overwriting
      specified keys) before execution.
  """

  model_config = pydantic.ConfigDict(extra="ignore")

  allow: bool = True
  message: str = ""
  modified_args: dict[str, Any] | None = None


class QuestionResponse(pydantic.BaseModel):
  """Individual response for an AskQuestion entry.

  Attributes:
    selected_option_ids: List of option IDs selected.
    freeform_response: Freeform text response.
    skipped: If true, the question is marked as skipped.
  """

  model_config = pydantic.ConfigDict(extra="ignore")

  selected_option_ids: list[str] | None = None
  freeform_response: str = ""
  skipped: bool = False


class QuestionHookResult(pydantic.BaseModel):
  """Result of an interaction containing a list of responses.

  Attributes:
    responses: List of QuestionResponse objects.
    cancelled: If true, the interaction was cancelled.
  """

  model_config = pydantic.ConfigDict(extra="ignore")

  responses: list[QuestionResponse]
  cancelled: bool = False


class AskQuestionOption(pydantic.BaseModel):
  """Option for an AskQuestion entry."""

  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  id: str
  text: str


class AskQuestionEntry(pydantic.BaseModel):
  """A single question with predefined options."""

  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  question: str
  options: list[AskQuestionOption]
  is_multi_select: bool = False


class AskQuestionInteractionSpec(pydantic.BaseModel):
  """Interaction spec for ask_question dialog."""

  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  questions: list[AskQuestionEntry]


class StopDecision(str, enum.Enum):
  """Decision returned by a Stop lifecycle hook.

  Attributes:
    ALLOW_STOP: Allows the turn execution to terminate and transition to
      STATE_FULLY_IDLE.
    CONTINUE: Blocks termination, injects reason as a system prompt, and resumes
      the agent execution loop.
  """

  ALLOW_STOP = "ALLOW_STOP"
  CONTINUE = "CONTINUE"


class StopHookResult(pydantic.BaseModel):
  """Result returned by a Stop lifecycle hook.

  Attributes:
    decision: Whether to allow the turn to stop or continue execution.
    reason: The prompt/feedback injected into the conversation when decision is
      CONTINUE. Must be non-empty when CONTINUE is selected; otherwise, raises a
      ValueError. Delivered directly to the model as a system message.
  """

  model_config = pydantic.ConfigDict(extra="ignore")

  decision: StopDecision = StopDecision.ALLOW_STOP
  reason: str = ""

  @pydantic.model_validator(mode="after")
  def _validate_continue_reason(self) -> "StopHookResult":
    if self.decision == StopDecision.CONTINUE and (
        not self.reason or not self.reason.strip()
    ):
      raise ValueError(
          "StopHookResult with decision=CONTINUE requires a non-empty reason."
      )
    return self


class StopArgs(pydantic.BaseModel):
  """Arguments delivered to a Stop hook when the root turn reaches idle.

  Attributes:
    response_text: Most recent assistant response text in the turn.
    trajectory_id: Unique identifier of the trajectory executing this turn.
    continuation_count: The 0-based iteration count of Stop hook continuations
      within the current turn cycle.
    stop_reason: The reason why the trajectory stopped (SDK StopReason enum
      value).
    error_message: Error message if execution stopped due to a fatal error.
  """

  model_config = pydantic.ConfigDict(extra="ignore")

  response_text: str = ""
  trajectory_id: str = ""
  continuation_count: int = 0
  stop_reason: StopReason = StopReason.UNSPECIFIED
  error_message: str = ""


# =============================================================================
# Error types
# =============================================================================


class AntigravityConnectionError(Exception):
  """Base class for connection errors in the Google Antigravity SDK.

  Raised when a connection to an agent backend cannot be established or
  encounters a fatal protocol-level error.
  """


class AntigravityCancelledError(asyncio.CancelledError):
  """Raised when an active turn is cancelled programmatically."""

  def __init__(self, message: str = "The request was cancelled by the client."):
    """Initializes the cancellation error with a default message."""
    super().__init__(message)


class AntigravityExecutionError(Exception):
  """Raised when the agent execution encounters a terminal error.

  This indicates that the agent loop has terminated due to a fatal error
  (e.g. model call failure, system constraint violation) and cannot continue.
  """


class ToolExecutionError(RuntimeError):
  """Raised when a tool execution fails, carrying tool metadata."""

  tool_name: str
  server_name: str | None
  call_id: str | None
  step_id: str | None

  def __init__(
      self,
      message: str,
      tool_name: str,
      server_name: str | None = None,
      call_id: str | None = None,
      step_id: str | None = None,
  ):
    super().__init__(message)
    self.tool_name = tool_name
    self.server_name = server_name
    self.call_id = call_id
    self.step_id = step_id


class AntigravityValidationError(Exception):
  """Wraps Pydantic ValidationError at the SDK boundary.

  SDK consumers should catch this instead of pydantic.ValidationError directly.
  This decouples the public API from the Pydantic implementation detail.

  Attributes:
    message: Human-readable error description.
    errors: The structured error list from Pydantic, if available.
  """

  def __init__(
      self,
      message: str,
      errors: list[dict[str, Any]] | None = None,
  ):
    super().__init__(message)
    self.message = message
    self.errors = errors or []

  @classmethod
  def _from_pydantic(
      cls, exc: pydantic.ValidationError
  ) -> "AntigravityValidationError":
    """Constructs from a Pydantic ValidationError.

    Args:
      exc: The original Pydantic ValidationError.

    Returns:
      An AntigravityValidationError wrapping the Pydantic error.
    """
    return cls(message=str(exc), errors=cast(Any, exc.errors()))


# =============================================================================
# Response types
# =============================================================================


class StreamChunk(pydantic.BaseModel):
  """Base class for all real-time semantic chunks yielded during agent.chat() streaming."""

  step_index: int
  model_config = pydantic.ConfigDict(frozen=True)


class Thought(StreamChunk):
  """A delta chunk representing a piece of the model's internal reasoning/thinking."""

  text: str  # Incremental thought string delta
  signature: bytes | None = None


class Text(StreamChunk):
  """A delta chunk representing a piece of the model's text output."""

  text: str  # Incremental response string delta


class ChatResponse:
  """The turn response from Agent.chat().

  An async stream of semantic chunks with lazy buffering.  Provides both
  zero-boilerplate text token streaming and advanced sugared event streams.

  Every iterator (``.chunks``, ``.thoughts``, ``.tool_calls``,
  ``async for delta in response``) returns an **independent cursor** over a
  shared buffer.  Cursors are safe to consume sequentially, or concurrently
  via ``asyncio.gather``.  If the upstream stream raises, the error is
  stored and re-raised to every cursor that reaches the end of the buffer.
  """

  def __init__(
      self,
      chunk_stream: AsyncIterator[StreamChunk | ToolCall | ToolResult],
      conversation: Any,
  ):
    self._chunk_stream = chunk_stream
    self._conversation = conversation
    self._buffered_chunks: list[StreamChunk | ToolCall | ToolResult] = []
    self._is_done = False
    self._stream_error: BaseException | None = None
    self._pull_lock = asyncio.Lock()

  @property
  def chunks(self) -> AsyncIterator[StreamChunk | ToolCall | ToolResult]:
    """The rich, unfiltered semantic chunk stream for more advanced use cases.

    Each call returns an **independent cursor** over the shared chunk buffer.
    Multiple cursors can be consumed sequentially or concurrently — each
    advances at its own pace.  When a cursor reaches the live edge of the
    buffer it pulls from the underlying network stream, appending chunks
    that all other cursors will also see.

    Concurrent safety is guaranteed by an internal lock that serializes
    network pulls — only one cursor awaits the upstream ``__anext__`` at a
    time.  If the upstream raises, the error is stored and re-raised to
    every cursor that reaches the end of the buffer.
    """

    async def _chunks_gen() -> (
        AsyncIterator[StreamChunk | ToolCall | ToolResult]
    ):
      pos = 0
      while True:
        if pos < len(self._buffered_chunks):
          yield self._buffered_chunks[pos]
          pos += 1
        elif self._is_done:
          if self._stream_error is not None:
            raise self._stream_error
          return
        else:
          async with self._pull_lock:
            # Re-check after acquiring — another cursor may have pulled
            # while we waited for the lock.
            if pos < len(self._buffered_chunks) or self._is_done:
              continue
            try:
              chunk = await self._chunk_stream.__anext__()
            except StopAsyncIteration:
              self._is_done = True
            except (asyncio.CancelledError, Exception) as e:
              self._is_done = True
              self._stream_error = e
              raise
            else:
              self._buffered_chunks.append(chunk)

    return _chunks_gen()

  async def __aiter__(self) -> AsyncIterator[str]:
    """Streams conversational text token deltas directly as raw strings."""
    async for chunk in self.chunks:
      if isinstance(chunk, Text):
        yield chunk.text

  @property
  def thoughts(self) -> AsyncIterator[str]:
    """The internal model reasoning/thinking token deltas as raw strings."""

    async def _thoughts_gen():
      async for chunk in self.chunks:
        if isinstance(chunk, Thought):
          yield chunk.text

    return _thoughts_gen()

  @property
  def tool_calls(self) -> AsyncIterator[ToolCall]:
    """The strongly-typed ToolCall objects in real-time as they are dispatched."""

    async def _tool_calls_gen():
      async for chunk in self.chunks:
        if isinstance(chunk, ToolCall):
          yield chunk

    return _tool_calls_gen()

  async def resolve(self) -> list[StreamChunk | ToolCall | ToolResult]:
    """Drains the underlying stream completely and returns all chunks as a flat list.

    Returns:
        A list of all chunks yielded during the turn.
    """
    return [chunk async for chunk in self.chunks]

  async def text(self) -> str:
    """Drains the stream and returns the fully aggregated conversational response text.

    Returns:
        The complete response text as a single string.
    """
    chunks = await self.resolve()
    return "".join(chunk.text for chunk in chunks if isinstance(chunk, Text))

  async def structured_output(self) -> Any | None:
    """Drains the stream and extracts the parsed structured output payload, if one exists.

    Returns:
        The parsed structured output if available, otherwise None.
    """
    if not self._is_done:
      await self.resolve()
    return self._conversation.get_last_structured_output()

  @property
  def usage_metadata(self) -> UsageMetadata | None:
    """Accumulated token usage across all model invocations in this turn."""
    return self._conversation.last_turn_usage

  @property
  def stop_reason(self) -> StopReason:
    """The reason why the execution turn stopped."""
    return self._conversation._last_turn_stop_reason  # pylint: disable=protected-access

  async def cancel(self) -> None:
    """Cancels the active execution turn and halts generation.

    This cleanly aborts the active stream on the backend. If the stream is
    already completed, this method acts as a safe no-op.
    """
    if not self._is_done:
      await self._conversation.cancel()


# =============================================================================
# Input Content Primitives
# =============================================================================

SUPPORTED_IMAGE_MIMES = frozenset({
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/webp",
})

SUPPORTED_DOCUMENT_MIMES = frozenset({
    "application/pdf",
    "application/json",
    "text/css",
    "text/csv",
    "text/html",
    "text/javascript",
    "text/plain",
    "text/rtf",
    "text/xml",
})

SUPPORTED_AUDIO_MIMES = frozenset({
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "audio/mp3",
    "audio/mp4",
    "audio/webm",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
    "audio/opus",
    "audio/mpeg",
    "audio/m4a",
    "audio/l16",
})

SUPPORTED_VIDEO_MIMES = frozenset({
    "video/3gpp",
    "video/avi",
    "video/mp4",
    "video/mpeg",
    "video/mpg",
    "video/quicktime",
    "video/webm",
    "video/wmv",
    "video/x-flv",
})


def _read_file_safely(path: str | pathlib.Path) -> bytes:
  """Robustly loads local file bytes with comprehensive error wrapping.

  Args:
      path: The file path to read.

  Returns:
      The file contents as bytes.

  Raises:
      FileNotFoundError: If the file does not exist.
      IsADirectoryError: If the path is a directory.
      PermissionError: If the file is not readable.
      OSError: For other filesystem errors.
  """
  file_path = pathlib.Path(path)
  try:
    return file_path.read_bytes()
  except FileNotFoundError as exc:
    raise FileNotFoundError(f"File not found at path: '{file_path}'") from exc
  except IsADirectoryError as exc:
    raise IsADirectoryError(
        f"Path is a directory, not a file: '{file_path}'"
    ) from exc
  except PermissionError as exc:
    raise PermissionError(
        f"Permission denied when reading path: '{file_path}'"
    ) from exc
  except OSError as exc:
    raise OSError(f"Failed to read file at path '{file_path}': {exc}") from exc


def _read_file_and_guess_mime(
    path: str | pathlib.Path,
) -> tuple[bytes, str]:
  """Reads a file and guesses its MIME type.

  Args:
      path: The file path to read.

  Returns:
      A tuple containing the file contents as bytes and the guessed MIME type
      as a string.

  Raises:
      ValueError: If the MIME type cannot be guessed.
  """
  file_path = pathlib.Path(path)
  data = _read_file_safely(file_path)
  mime_guess, _ = mimetypes.guess_type(file_path)
  if not mime_guess:
    raise ValueError(
        f"Could not infer a valid MIME type for extension: '{file_path.suffix}'"
    )
  return data, mime_guess


class _BaseMedia(pydantic.BaseModel):
  """Base class for all rich multimedia content attachment primitives."""

  _SUPPORTED_MIMES: ClassVar[frozenset[str]] = frozenset()

  data: bytes
  mime_type: str
  description: str | None = None

  @pydantic.field_validator("mime_type")
  @classmethod
  def _validate_mime_type(cls, v: str) -> str:
    """Validates that the MIME type is supported for this media type."""
    if cls is _BaseMedia:
      return v
    if v not in cls._SUPPORTED_MIMES:
      raise ValueError(f"Unsupported {cls.__name__} MIME type: '{v}'")
    return v

  @classmethod
  def from_file(
      cls: type[_BaseMediaT],
      path: str | pathlib.Path,
      description: str | None = None,
  ) -> _BaseMediaT:
    """Instantiates a media content primitive from a local file path.

    Args:
        path: Local file path to read.
        description: Optional text description of the media.

    Returns:
        The instantiated media object.
    """
    data, mime_type = _read_file_and_guess_mime(path)
    return cls(
        data=data,
        mime_type=mime_type,
        description=description,
    )

  model_config = pydantic.ConfigDict(frozen=True)


class Image(_BaseMedia):
  """Image content attachment primitive."""

  _SUPPORTED_MIMES: ClassVar[frozenset[str]] = SUPPORTED_IMAGE_MIMES


class Document(_BaseMedia):
  """Document content attachment primitive."""

  _SUPPORTED_MIMES: ClassVar[frozenset[str]] = SUPPORTED_DOCUMENT_MIMES


class Audio(_BaseMedia):
  """Audio content attachment primitive."""

  _SUPPORTED_MIMES: ClassVar[frozenset[str]] = SUPPORTED_AUDIO_MIMES


class Video(_BaseMedia):
  """Video content attachment primitive."""

  _SUPPORTED_MIMES: ClassVar[frozenset[str]] = SUPPORTED_VIDEO_MIMES


class BuiltinSlashCommandName(str, enum.Enum):
  """Supported system slash commands.

  Attributes:
    PLAN: Plan carefully before executing a task (generates an implementation
      plan artifact and awaits user approval).
  """

  PLAN = "plan"


class SlashCommand(pydantic.BaseModel):
  """Slash command context primitive.

  Attributes:
    name: The strict BuiltinSlashCommandName enum.
  """

  model_config = pydantic.ConfigDict(frozen=True)

  name: BuiltinSlashCommandName


ContentPrimitive = str | Image | Document | Audio | Video | SlashCommand
Content = ContentPrimitive | Sequence[ContentPrimitive]

# Registry mapping each supported MIME type to its media class.
# Built once at import time from the per-category frozensets.
_MIME_TO_MEDIA_CLASS: dict[str, type[_BaseMedia]] = {
    mime: cls
    for mime_set, cls in [
        (SUPPORTED_IMAGE_MIMES, Image),
        (SUPPORTED_DOCUMENT_MIMES, Document),
        (SUPPORTED_AUDIO_MIMES, Audio),
        (SUPPORTED_VIDEO_MIMES, Video),
    ]
    for mime in mime_set
}


def from_file(
    path: str | pathlib.Path, description: str | None = None
) -> Image | Document | Audio | Video:
  """Automatically resolves a local file path into the correct semantic Content primitive.

  Args:
      path: Local file path to read.
      description: Optional text description of the media.

  Returns:
      A specialized media object (Image, Document, Audio, or Video) based
      on the file's MIME type.

  Raises:
      ValueError: If the MIME type cannot be inferred or is unsupported.
  """
  data, mime_type = _read_file_and_guess_mime(path)
  return from_bytes(data, mime_type, description)


def from_bytes(
    data: bytes, mime_type: str, description: str | None = None
) -> Image | Document | Audio | Video:
  """Automatically resolves raw bytes and a MIME type into the correct Content primitive.

  Args:
      data: Raw file bytes.
      mime_type: The MIME type of the content.
      description: Optional text description of the media.

  Returns:
      A specialized media object (Image, Document, Audio, or Video) based on the
      MIME type.

  Raises:
      ValueError: If the MIME type is unsupported.
  """
  media_cls = _MIME_TO_MEDIA_CLASS.get(mime_type)
  if media_cls is None:
    raise ValueError(
        f"Unsupported MIME type: '{mime_type}'. "
        f"Supported file formats in the SDK are: {sorted(_MIME_TO_MEDIA_CLASS)}"
    )
  return cast(
      Image | Document | Audio | Video,
      media_cls(data=data, mime_type=mime_type, description=description),
  )
