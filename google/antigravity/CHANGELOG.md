# Changelog




All notable changes to the Google Antigravity Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.16] - 2026-08-31

This release updates the default model for new agents to `gemini-3.8-flash` for higher quality and reasoning capabilities, alongside improving agent configuration expressiveness, performance for small and local models, and platform connectivity. Developers can now easily configure lightweight agents optimized for local environments via a new fluent method, connect to Vertex AI with API keys in Express mode, and benefit from expanded support for custom tool implementations (functors, dataclasses).

### 🌟 Key Highlights
- **Default Model Update to Gemini 3.8 Flash**: The default model for new agents and configurations has been updated to `gemini-3.8-flash`, delivering higher reasoning quality and stronger task performance. Developers can override this default by specifying `model` in their configuration:
  ```python
  from antigravity.connections.local import LocalAgentConfig

  config = LocalAgentConfig(model="gemini-3.7-flash")
  ```

- **Optimized Lightweight Agent Configuration**: New `.lightweight()` method on agent configurations (including `LocalAgentConfig`) and its subclasses applies preset optimizations for smaller, local-running models. This configures a minimal tool set, minimal system prompting, and disables background subagents to reduce context exhaustion and latency.
  ```python
  import os
  from antigravity.connections.local import LiteRTAgentConfig

  config = LiteRTAgentConfig(
      model_path=os.path.expanduser(
          "~/.litert-lm/models/gemma4-26b/model.litertlm"
      )
  ).lightweight()
  ```

- **Vertex AI Express Mode (API Key Support)**: Developers can now connect to Vertex AI using Express mode by providing an API key directly on `LocalAgentConfig(vertex=True, api_key="...")`, simplifying authentication without needing full GCP project/location ADC setup.
  ```python
  from antigravity.connections.local import LocalAgentConfig

  # Express Mode using API Key
  config = LocalAgentConfig(vertex=True, api_key="AIza...")
  ```

- **Support for Callable Class and Dataclass Tools**: The SDK's tool runner, `ToolRunner`, now correctly inspects and executes tools defined as callable class instances (functors) and dataclass methods, enhancing flexibility for custom tool implementation.
  ```python
  class MyTool:
    def __call__(self, context: ToolContext, *args):
        # ... tool logic
        pass

  agent.config.add_tool(MyTool())
  ```

- **Stop Hook Integration**: Agents now support the `StopHook` lifecycle hook, which is triggered when an agent's execution is externally requested to stop. This enables developers to implement custom cleanup, messaging, or resource logging actions upon termination.
  ```python
  class CleanupHook(hooks.StopHook):
    def on_stop(self, agent: Agent):
      print("Agent stopped. Performing cleanup.")
  ```

### 📋 Detailed Changes

#### Model & Default Changes
- **Default Model Update**: The default model for new agents and configurations is now `gemini-3.8-flash`. This change provides higher output quality and reasoning capabilities. To override this default, simply specify the `model` parameter during agent configuration:
  ```python
  from antigravity.connections.local import LocalAgentConfig
  
  config = LocalAgentConfig(model="gemini-3.7-flash")
  ```

#### Features & Enhancements
- **OS Sandbox Opt-in for Commands**: Added an `enable_sandbox` field to `RunCommandConfig` to allow developers to optionally execute terminal commands within an OS-level sandbox environment for enhanced security.
  ```python
  from antigravity.types import RunCommandConfig
  
  config.capabilities.run_command_config = RunCommandConfig(
      enable_sandbox=True
  )
  ```
- **Tool Invocation Argument Flexibility**: `ToolWithSchema` and public callable proxies now accept both positional (`*args`) and keyword arguments (`**kwargs`) when called, matching standard Python function calling conventions.
- **Support for `genai.Content` Media**: Introduced a structconverter to support media blocks from `genai.Content` objects within the SDK, enabling richer multimodal interactions.
- **MCP Compatibility Widening**: The SDK now supports both `mcp>=1.0` and `mcp<3.0` dependencies, ensuring compatibility with new and existing MCP environments.

#### Bug Fixes & Deprecations
- **Agent Behavior Fix for Interactive Examples**: Corrected an issue where interactive SDK examples (`interactive_cli.py`, `human_in_the_loop.py`, etc.) were not consistently configured with `AgentBehavior.INTERACTIVE`, ensuring proper question asking and hook activation.
- **Tool Runner Execution Environment Fix**: Resolved a failure in exported SDK examples (e.g., in CI environments) where `stdio` MCP servers were executed improperly. Child processes now correctly use `sys.executable` to run within the active Python virtual environment.

- **DEPRECATION: Step Token Usage**: `Step.usage_metadata` has been deprecated. Usage reporting is now consolidated at the turn-level (`ChatResponse.usage_metadata`) and session-level (`agent.conversation.total_usage`) to provide a more accurate metric, as a single model invocation often maps to multiple execution steps.

## [0.1.15] - 2026-08-25

Antigravity Python SDK v0.1.15 introduces subagent-exclusive tool scoping to reduce context overhead, adds an `on_compaction` lifecycle hook for observing context checkpointing, expands platform compatibility with Alpine Linux musl wheels, resolves workspace path normalization and custom Vertex endpoint routing, and adds universal JSON Schema normalization for custom Python tools when targeting local OpenAI-compatible LLM endpoints.

### 🌟 Key Highlights
- **Subagent-Scoped Custom Tools**: Custom tools can now be registered directly on subagents without requiring registration on the root agent. Subagent-specific tools are strictly isolated to the subagent's execution context and will not leak into the root agent's prompt or consume root context tokens.
- **Context Compaction Lifecycle Hook**: Improved support for the `on_compaction` lifecycle hook to more precisely capture compaction events and summaries whenever long-running conversations trigger checkpointing.
- **Custom Base URL & Secure Vertex Proxy Support**: `VertexEndpoint` now supports routing requests to custom `base_url` reverse proxies or enterprise gateways without leaking ambient Google Cloud Application Default Credentials (ADC) OAuth tokens or conflicting with project/location configurations.
- **Universal JSON Schema Normalization for Custom Tools**: Custom Python tools now produce standard OpenAPI / JSON Schema compliant parameter definitions with lowercase types and camelCase combiners, eliminating HTTP 400 Bad Request schema errors when connecting to local OpenAI-compatible engines such as Ollama, LM Studio, or vLLM.

---

### 📋 Detailed Changes

#### Features & Enhancements
- **`from_bytes` Helper Export**: Exported the `from_bytes` helper in top-level `google.antigravity` alongside `from_file` for creating binary/multimodal content payloads.
- **PEP 656 musllinux Wheel Support**: Added `musllinux_1_1` wheel platform tags for x86_64 and aarch64 architectures, enabling direct installation via pip on Alpine Linux containers.
- **Isolated Harness Environment Configuration**: Added per-connection environment dictionary resolution for `ANTIGRAVITY_HARNESS_PATH`, avoiding the need to mutate global `os.environ`.
- **Tool Call Metadata Preservation**: Preserved `id`, `step_id`, and `server_name` metadata across all `ToolResult` executions, including batch calls, errors, and unknown tools.

#### Model & Default Changes
- **Relative Workspace Path Normalization**: `LocalAgentConfig(workspaces=...)` now resolves relative directory paths and tilde (`~`) expansions against the current working directory (`os.getcwd()`). Previously, relative paths were passed unresolved to the harness and interpreted as absolute root paths, causing workspace indexing failures. Pass absolute paths or `pathlib.Path` instances if referencing directories outside the current working tree.
- **Deprecated `TriggerDelivery` Removal**: Removed the unused `TriggerDelivery` enum from `types.py`.

#### Bug Fixes
- **Local OpenAI Tool Schema Validation**: Fixed HTTP 400 "Invalid discriminator value" errors on local OpenAI endpoints by canonicalizing tool parameter schemas to standard JSON Schema.
- **Vertex Custom Gateway Token Leaks**: Fixed host GCP OAuth bearer token leakage and environment variable collisions when using custom `base_url` endpoints with `VertexEndpoint`.
- **Subagent Custom Tool Routing**: Fixed tool dispatch failures when subagents defined tools not registered on the root agent.
- **`UsageMetadata` Service Tier Loss**: Fixed `UsageMetadata.__sub__` dropping the `service_tier` field when calculating per-turn usage deltas.
- **WebSocket Deprecation Warnings**: Resolved `DeprecationWarning` exceptions when reading WebSocket close codes across varying websockets library versions.
- **OpenTelemetry Optional Dependency Guard**: Fixed test collection failures on minimal environments lacking `opentelemetry.sdk`.

## [0.1.14] - 2026-08-21

Bug fixes:
- https://github.com/google-antigravity/antigravity-sdk-python/issues/183
- https://github.com/google-antigravity/antigravity-sdk-python/issues/181
- https://github.com/google-antigravity/antigravity-sdk-python/issues/167

## [0.1.13] - 2026-08-18

This release introduces pre-tool argument modification capabilities in lifecycle hooks, adds native support for synchronous hook functions, and establishes structured command execution configuration with configurable execution timeouts. It also improves tool execution observability with step correlation IDs, and enhances connection resilience during client disconnects.

### 🌟 Key Highlights
- **Pre-Tool Hook Argument Modification**: Pre-tool lifecycle hooks can now sanitize, transform, or override tool input arguments before tool execution begins.
  ```python
  @pre_tool
  def sanitize_args(event: PreToolHookEvent) -> PreToolHookDecision:
      return PreToolHookDecision(allow=True, modified_args={"query": event.args["query"].strip()})
  ```

- **Synchronous Hook Function Support**: Lifecycle hook decorators now accept standard synchronous functions alongside asynchronous coroutines without raising runtime await errors.
  ```python
  @pre_turn
  def log_turn(event: PreTurnHookEvent) -> None:
      print(f"Executing turn for session: {event.session_id}")
  ```

- **Structured Command Execution Configuration**: Command execution settings are now consolidated under `RunCommandConfig`, introducing configurable timeouts that default to 10 minutes (600 seconds) alongside daemon execution controls.
  ```python
  capabilities = CapabilitiesConfig(
      run_command_config=RunCommandConfig(timeout_seconds=300, enable_daemons=True)
  )
  ```

- **Tool Lifecycle Step Correlation**: `ToolResult` and `ToolExecutionError` event payloads now include `step_id`, enabling end-to-end tracking and correlation of tool invocations across trajectory steps.
  ```python
  @post_tool
  def trace_tool_execution(event: ToolResult) -> None:
      print(f"Step {event.step_id}: {event.tool_name} returned {event.result}")
  ```

---

### 📋 Detailed Changes

#### Features & Enhancements
- **Pre-Tool Hook Argument Modification**: Added the ability for `@pre_tool` hooks to return updated tool arguments that are sequentially chained across registered hooks prior to tool execution.
- **Tool Step Correlation**: Added `step_id` to `ToolResult` and `ToolExecutionError` hook event models, allowing developers to correlate tool completions and failures with their originating tool call step.
- **VS Code Debugging Configuration**: Updated `setup_vscode_debugging.sh` to target the canonical `getting_started/hello_world` starter example and explicitly configure Gemini Developer API defaults.

#### Model & Default Changes
- **Command Timeout and Daemon Configuration**: `CapabilitiesConfig` now has a `RunCommandConfig` allowing configuration of execution timeouts and daemon commands. Timeouts default to 10 minutes (600 seconds) to prevent unresponsive command tasks. Run commands can have daemons enabled via a boolean flag..

#### Bug Fixes
- **Synchronous Hook Decorator Execution**: Fixed a runtime `TypeError` when decorating synchronous functions with `@pre_turn`, `@post_tool_call`, and other lifecycle hooks by verifying awaitability before awaiting hook responses.
- **LiteRT Early Client Disconnects**: Suppressed unhandled `ConnectionResetError`, `ConnectionAbortedError`, and `BrokenPipeError` exceptions when clients disconnect early from local LiteRT server connections.

## [0.1.12] - 2026-08-13

This release fixes a regression: https://github.com/google-antigravity/antigravity-sdk-python/issues/172

## [0.1.11] - 2026-08-11

The 0.1.11 release updates the default model to `gemini-3.7-flash`, introduces session-level budget enforcement and turn termination stop reasons, Vertex AI Express Mode authentication and a new agent behavior setting that toggles betweein interactive and autonomous (the default). It also expands tool hook metadata, resolves string annotation coercion for postponed evaluation, and improves MCP server and subagent stability.

### 🌟 Key Highlights

- **Default Model Upgrade to Gemini 3.7 Flash**: Upgraded the default inference model to `gemini-3.7-flash`.
- **Session Budget Enforcement & Stop Reasons**: Added `BudgetConfig` to define session-level usage limits (model invocations, tool invocations, and token budgets) and `StopReason` enum (`MAX_MODEL_CALLS_EXCEEDED`, `MAX_TOOL_CALLS_EXCEEDED`, `MAX_TOTAL_TOKENS_EXCEEDED`, `QUOTA_EXHAUSTED`, etc.) to inspect turn termination causes.
- **Vertex AI Express Mode Support**: Added native support for Express Mode authentication via `VertexEndpoint(api_key=...)` and `LocalAgentConfig(vertex=true, api_key=...)`, simplifying headless and non-GCP deployments.
- **Autonomous Agent Behavior Mode**: Control the agent's behavior with `AgentBehavior`. By default the SDK now has a `AgentBehavior.AUTONOMOUS` mode (used to be `AgentBehavior.INTERACTIVE`) to streamline scripting, background and headless interactions modes.
- **Multi-Interface Hook Registration**: Enabled single-instance registration across multiple hook interfaces (`PreToolHook`, `PostToolHook`, `PreTurnHook`), allowing for cross functional instrumentation.

### 🔧 Detailed Changes

#### New Features
- **Budget Enforcement**: Introduced `BudgetConfig(max_model_calls, max_tool_calls, max_input_tokens, max_output_tokens, max_total_tokens)` and exposed stop reason metadata on turn responses.
- **Express Mode API Key Auth**: Added `api_key` support across `VertexEndpoint` and local agent configurations.
- **Multi-Interface Hooks**: Supported registering composite hook classes without duplicate invocation.
- **PreToolArgs Metadata**: Exposed `step_id` on tool hook payloads for chat thread context tracing.

#### Model & Default Changes
- **Default Model Upgrade**: Updated the default model from `gemini-3.6-flash` to `gemini-3.7-flash`.
- **Agent Behavior**: Introduced the ability to control `AgentBehavior`. Now defaults to autonomous (form interactive) to avoid unexpected interactive pause states during script execution. Override by setting `CapabilitiesConfig(agent_behavior=AgentBehavior.INTERACTIVE)`.

#### Bug Fixes & Maintenance
- **ToolRunner String Annotation Coercion**: When using `from __future__ import annotations`, tool argument coercion failed on stringified types; resolved by resolving type annotations via `typing.get_type_hints` before type adaptation.
- **MCP Server Example Port Binding**: Ephemeral port race conditions in test and example server startup were resolved by binding directly to port 0.
- **Subagent Deadlock Prevention**: Handled subagent fatal errors in the localagent executor to prevent deadlocks when subagents terminate abnormally.
- **Empty Input Validation**: Added input validation to prevent SDK unresponsiveness on empty or whitespace-only prompts.

## [0.1.10] - 2026-08-04

The 0.1.10 release introduces support for Gemini Prioritized Inference service tiers, real-time token usage event streaming, stateful context-aware hook decorators, and explicit tool call correlation IDs across hook callbacks. It also standardizes default system instruction merging behavior, fixes WebSocket compaction events, and expands local Gemma model documentation.

### 🌟 Key Highlights

- **Gemini Prioritized Inference Service Tier**: Configure agents to utilize Gemini Prioritized Inference service tiers for high-priority model execution with automated graceful fallback.
  ```python
  from google.antigravity import (
      GeminiAPIEndpoint,
      GeminiModelOptions,
      LocalAgentConfig,
      ModelTarget,
      types,
  )

  # Configure priority inference via GeminiAPIEndpoint
  model_opts = GeminiModelOptions(
      service_tier=types.ServiceTier.PRIORITY,
  )
  config = LocalAgentConfig(
      model=ModelTarget(
          name="gemini-3.6-flash",
          endpoint=GeminiAPIEndpoint(options=model_opts),
      ),
  )
  ```

- **Tool Call ID Correlation in Lifecycle Hooks**: Inspect `call_id` attributes on tool executions, errors, and hooks to correlate multi-step tool invocations across lifecycle callbacks.
  ```python
  from google.antigravity import ToolExecutionError, types
  from google.antigravity.hooks import hooks

  @hooks.pre_tool_call_decide
  async def pre_tool(data: types.ToolCall) -> types.HookResult:
      print(f"Executing tool {data.name} (call_id={data.id})")
      return types.HookResult(allow=True)

  @hooks.on_tool_error
  async def on_error(error: ToolExecutionError) -> None:
      print(f"Tool {error.tool_name} failed (call_id={error.call_id}): {error}")
  ```

- **Context-Aware Hook Decorators**: Decorate hook handlers (`@hooks.pre_turn`, `@hooks.post_tool_call`, etc.) that optionally accept `HookContext` as a parameter to maintain state and share data across lifecycle callbacks.
  ```python
  from google.antigravity import types
  from google.antigravity.hooks import HookContext, hooks

  @hooks.pre_turn
  async def inspect_prompt(context: HookContext, data: str) -> types.HookResult:
      context.set_state("user_prompt", data)
      return types.HookResult(allow=True)
  ```

- **ActionCompaction Event Emission & Hook**: Track context window compaction notifications over WebSockets and intercept them using `@hooks.on_compaction`.
  ```python
  from google.antigravity.hooks import hooks

  @hooks.on_compaction
  async def handle_compaction(data) -> None:
      print(f"Context compaction occurred: {data}")
  ```

- **Standardized System Instructions Strategy**: Plain string instructions default to appending to built-in instructions. To override and completely replace built-in instructions, pass `CustomSystemInstructions`.
  ```python
  from google.antigravity import CustomSystemInstructions, LocalAgentConfig

  # Override and replace built-in instructions completely
  config = LocalAgentConfig(
      system_instructions=CustomSystemInstructions(
          text="You are a specialized code reviewer."
      )
  )
  ```

---

### 📋 Detailed Changes

- **Features & Enhancements**
  - **Tool Execution Call ID Correlation**: Added `call_id` attributes to `ToolCall.id`, `ToolResult.id`, `ToolExecutionError.call_id`, and hook payloads to enable tracing of tool execution steps.
  - **Live Token Usage Reporting**: Introduced real-time `UsageUpdate` event streaming so token usage accumulates live during agent execution rather than delaying updates until state transitions.
  - **Context-Aware Hook Decorators**: Enabled decorated hook handlers to optionally accept `HookContext` parameters for stateful hook implementations while preserving backward compatibility for stateless handlers.
  - **Interactive CLI Spinner**: Updated CLI interactive loop spinner to list all active tool names when running concurrent tool calls (e.g., `Running tools 'tool_a', 'tool_b'`).
  - **Module Re-exports**: Re-exported `ReadUrlContentResult` and `SearchWebResult` in `connections.local` for uniform tool result access.
  - **Local Gemma Model Documentation**: Added guides and tutorials for running agents locally with Gemma models using LiteRT and OpenAI-compatible endpoints.

- **Model & Default Changes**
  - **System Instruction Append Policy**: Standardized string system instructions across all connection types to default to an append strategy (combining custom string instructions with built-in instructions). To override this default and completely replace built-in instructions, pass `CustomSystemInstructions(text=...)`.
  - **LiteRT Token Output Limit**: Increased `max_output_tokens` default in LiteRT local server configuration from 8,192 to 16,384 tokens to prevent truncation during complex reasoning and generation tasks.

- **Bug Fixes**
  - **ActionCompaction Event Emission**: Fixed issue where compaction notifications were suppressed in external SDK releases, causing `@hooks.on_compaction` handlers and `conversation.compaction_indices` tracking to fail; compaction events now emit properly over WebSockets.
  - **MCP Test Server Startup**: Fixed a race condition where the HTTP port was exposed before uvicorn server startup completed, which previously caused intermittent `ConnectionRefusedError` failures during test initialization.

## [0.1.9] - 2026-07-27

Release 0.1.9 of the Google Antigravity Python SDK adds model-call retry/backoff configuration, improves tool registration for custom functions, adds missing `BuiltinTools` exports, fixes user audio payload processing, and adds connection `DebugConfig` support along with `ToolExecutionError` handling.

### 🌟 Key Highlights

- **Model-Call Retry & Backoff Configuration**: Exposes configurable retry and backoff parameters for model calls.
- **LiteRT Warm-up Timeout Scaling**: Dynamically scales LiteRT engine warm-up timeouts based on context size and synchronizes warm-up cleanup.
- **Improved Tool Registration**: Enhances automatic tool registration for custom Python functions.
- **BuiltinTools Typing Exports**: Exports `BuiltinTools` at top-level typing boundaries for subagent tool configuration.
- **Audio Payload Processing Fix**: Fixes processing for user audio payloads in interactive agent sessions.
- **ToolExecutionError Handling**: Introduces `ToolExecutionError` for explicit tool failure handling.
- **Connection DebugConfig**: Adds base `DebugConfig` options to Connection configuration.

---

### 📋 Detailed Changes

#### Features & Enhancements
- **Retry & Backoff**: Expose model-call retry/backoff configuration options in LocalAgentConfig.
- **Tool Registration**: Improve automatic tool registration and docstring parsing for custom functions.
- **BuiltinTools**: Export `BuiltinTools` from main package namespace.
- **DebugConfig**: Add `DebugConfig` for enhanced connection debugging and logging.
- **Error Handling**: Add `ToolExecutionError` to SDK error types.

#### Bug Fixes
- **Audio Payloads**: Fix user audio payload processing in agent sessions.

## [0.1.8] - 2026-07-21

Release 0.1.8 of the Google Antigravity Python SDK updates the default text model to Gemini 3.6 Flash, adds support for custom subagent instructions, and implements automatic Pydantic argument coercion for tool calls. It also brings defensive prompt sanitization, configurable tool retries, and comprehensive stability improvements for local agent execution.

### 🌟 Key Highlights

- **Default Model Upgrade to Gemini 3.6 Flash**: Upgrades the default generative text model in the Python SDK to `gemini-3.6-flash`.
  ```python
  from google.antigravity import LocalAgentConfig, models

  # Defaults to models.DEFAULT_MODEL ("gemini-3.6-flash")
  config = LocalAgentConfig()
  ```

- **Pydantic TypeAdapter Tool Argument Coercion**: Automatically coerces stringified numbers, booleans, and nested models returned by LLMs into strict Python types declared in tool signatures.
  ```python
  def calculate_scale(factor: int, active: bool = True) -> float:
      return factor * 1.5 if active else 0.0
  # Automatically converts {"factor": "5", "active": "true"} to int and bool
  ```

- **Custom Subagent Instructions**: Adds capability to specify custom system instructions for spawned subagents in multi-agent workflows.
  ```python
  from google.antigravity import LocalAgentConfig, SubagentConfig

  config = LocalAgentConfig(
      subagents=[SubagentConfig(name="researcher", system_instructions="You are a research specialist.")]
  )
  ```

- **Prompt Sanitization**: Strips null bytes and non-printable control characters from incoming user prompts at the wire boundary to prevent HTTP 400 errors and terminal corruption.

---

### 📋 Detailed Changes

#### Features & Enhancements
- **Custom Subagent Instructions**: Allow subagents to receive isolated system instructions and allowlisted tool configurations during execution.
- **Pydantic Argument Coercion**: Use `pydantic.TypeAdapter` in `ToolRunner` to seamlessly validate and parse `Optional`, `Union`, list, and primitive tool arguments.
- **Prompt Sanitization**: Automatically strip control codes (`DEL`, `BEL`, `C1`) and null bytes (`\x00`) from user inputs.
- **Configurable Tool Retries**: Add `RetryConfig` definitions in local configuration to allow fine-tuned model and output retry policies.
- **Operating System Telemetry**: Populate client OS and version information in telemetry headers to improve diagnostic tracking.
- **Native Usage Accumulation**: Enable native addition (`+` and `+=`) operations on `UsageMetadata` instances.

#### Model & Default Changes
- **Default Text Model**: Update default text model from `gemini-1.5-flash` to `gemini-3.6-flash` across SDK interfaces. To override, specify `model="gemini-1.5-pro"` (or your preferred endpoint) in `LocalAgentConfig`.

#### Bug Fixes & Stability
- **Large Tool Output Handling**: Dynamic output truncation and removal of WebSocket frame limits to resolve connection termination on large tool responses.
- **Subagent Idle Synchronization**: Fix race conditions where multiple idle states could cause `receive_steps()` to hang indefinitely.
- **Custom Tool Policy Enforcement**: Ensure custom tools properly trigger pre-tool policy checks and cleanly report denials.
- **Media MIME Type Inference**: Raise `ValueError` instead of Pydantic validation failures when media MIME types cannot be inferred from file extensions.
- **LiteRT Log Noise**: Suppress verbose C++ LiteRT engine diagnostic output by default.
- **Policy Copy Idempotency**: Prevent duplicate workspace policy prepending during deep copies of `BaseLocalAgentConfig`.
- **Type Safety**: Enforce typed `types.Step` arguments in `OnCompactionHook`.

## [0.1.7] - 2026-07-14

Release 0.1.7 of the Google Antigravity Python SDK expands end-user control over agent execution environments, strengthens concurrency safety for stateful tools, and introduces deeper reasoning capabilities. Key highlights include atomic multi-threaded state handling for tools and hooks, customizable subprocess environment variable isolation, support for an "extra_high" thinking severity level, and full Model Context Protocol (MCP) and subagent support across local model backends. This release also resolves interactive console prompt clobbering and improves socket discovery under containerized setups.

### 🌟 Key Highlights
- **Multi-threaded Hook & Tool State Handling**: Developers can now safely read and mutate shared context variables across multi-threaded tools (`asyncio.to_thread` or `ThreadPoolExecutor`) using atomic updates and thread locking:
  ```python
  def my_tool(ctx: ToolContext) -> str:
      with ctx.lock():
          ctx.update_state("count", lambda c: (c or 0) + 1)
      return "Success"
  ```
- **Custom Subprocess Environment Variables**: Developers can now pass custom environment variables directly to isolated agent instances via `LocalAgentConfig`, avoiding pollution of the global parent environment:
  ```python
  config = LocalAgentConfig(env={"PATH": "/custom/bin", "MY_API_KEY": "secret"})
  ```
- **Extra High Thinking Severity Support**: Developers can now configure an `"extra_high"` thinking severity level for complex reasoning tasks without needing to specify or override the model name:
  ```python
  config = LocalAgentConfig(
      model=ModelTarget(
          endpoint=GeminiAPIEndpoint(
              options=GeminiModelOptions(thinking_level="extra_high")
          )
      )
  )
  ```
- **MCP & Subagent Support for Local Models (LiteRT & OpenAI)**: Developers using local Gemma (`LiteRTAgentConfig`) or local OpenAI-compatible endpoints (`LocalOpenAIAgentConfig`, such as Ollama or LM Studio) can now directly configure subagents and register Model Context Protocol (MCP) servers, enabling full local multi-agent and MCP tool workflows.

---

### 📋 Detailed Changes

#### Features & Enhancements
- **Environment Hydration**: Hydrates GCP/Vertex parameters (project, location, routing) dynamically from standard GOOGLE_CLOUD environment variables when not explicitly passed during LocalAgentConfig initialization.
- **Default Image Generation Model Optimization**: Updated the default image generation model (`DEFAULT_IMAGE_GENERATION_MODEL`) to `"gemini-3.1-flash-lite-image"`. Previous default image models often took too long to run on average during standard agent execution loops; this lightweight model ensures dependable, high-speed image generation by default while remaining fully replaceable via explicit model configuration if higher fidelity is required.

#### Robustness & Usability
- **Local WebSocket Connections**: Retries socket connections by resolving to both "localhost" and "127.0.0.1", ensuring dependable harness discovery under containerized setups.
- **Tool Runner Public Asyncness**: Preserves original asynchronous and synchronous execution interfaces of tools when accessed via ToolRunner.get_public_callable.
- **Config Inheritance and Gaps**: Cleaned up redundant base overrides in LocalAgentConfig and corrected Pydantic validation failures arising from None initialization variables.

#### Bug Fixes
- **Interactive Console Spinner Clobbering**: Fixed a bug in `run_interactive_loop` where background spinner animation frames (`⠼ Reasoning...`) continuously clobbered user confirmation prompts (`async_input` and `ASK_USER` policy checks) every 80ms. The active spinner is now explicitly cleared (`\r\033[K`) and paused when an interactive input prompt opens, keeping confirmation lines readable and resuming the spinner smoothly after input is submitted.
- **Duplicate Tool Call Events**: Fixed client rendering issues by filtering out custom tool events from StepUpdate payloads to prevent duplicate event dispatches.
- **SDK Idle Transitions**: Corrected premature agent shutdown situations by properly checking and clearing idling flags once a TrajectoryStateUpdate signals transition to running.
- **LiteRT Connection Engine**: Rectified local engine initialization bugs, including a missing protobuf import, a token constructor argument mismatch, and OpenAITool inheritance.
- **Shutdown Connection Handshake**: Added extra execution buffer to local connections during agent shutdown, ensuring safe persistent state storage actions.

## [0.1.6] - 2026-07-09

This release expands local execution capabilities by broadening support for multimodal and web-connected tool workflows. Developers can now run Gemma models locally using LiteRT or integrate with local OpenAI-compatible APIs, natively return multimodal media from custom tools, and leverage a more robust, streamlined lifecycle hooks framework that is fully aligned with Model Context Protocol (MCP) workflows.

### Added
- **Local Model Connectivity**: Introduced `LiteRTAgentConfig` and `LiteRTConnectionStrategy` for LiteRT-LM (supporting local Gemma execution), `LocalOpenAIAgentConfig` and `LocalOpenAIConnectionStrategy` for OpenAI-compatible APIs (supporting Ollama and LM Studio), and a background loopback HTTP translation server.
- **Multimodal Tool Outputs**: Enabled custom tools to return media assets (`Image`, `Document`, `Audio`, `Video`) directly via a single tool response without needing separate follow-up turns (`supplemental_media`).
- **Built-in Web Fetch Tool**: Integrated the `read_url_content` tool end-to-end for fetching structured web content natively with the `ReadUrlContentResult` Pydantic model.

### Changed
- **MCP String Prefix Modernization**: Decoupled tool calls and safety policy engines from legacy `"mcp_"` string synthesis, resolving name mismatch issues by utilizing explicit `server_name` attributes in tool evaluation.

### Fixed
- **Python 3.14 Compatibility**: Resolved namespace class conflicts and typing normalization issues in `agent.py` and `public_api_test.py` under Python 3.14 deferred annotations evaluation.
- **Vertex Validation Errors**: Cleared a misleading reference to API keys in `VertexEndpoint` validation error messages, limiting fields to project and location.
- **OTel Trace Warnings**: Resolved detached `contextvars` warnings and set-status race conditions by removing `use_span` context managers from Turn/Session hooks and checking span recording readiness.
- **Exception Wrapping Mapping**: Fixed `agent_middleware` integration check failures by making the example check for error message substrings instead of strict exception types.

---

## [0.1.5] - 2026-06-25

This release introduces native OpenTelemetry tracing support, declarative subagent configurations, improved type safety, and critical robustness and compatibility updates.

### Added
- **OpenTelemetry Tracing Support**: Integrates OpenTelemetry tracing into the SDK to translate session, turn, step, and tool lifecycle events into standard GenAI-compliant semantic spans for advanced monitoring and performance debugging, with custom task-safe active span propagation for tool execution.
- **Declarative Subagent Configurations**: Added `SubagentConfig` and `SubagentCapabilities` in `types.py` to support constructing static subagents with declarative instructions and tools directly.

### Changed
- **Type Safety in `AgentConfig`**: Type-annotated policies, hooks, and triggers parameters on `AgentConfig` and its subclasses to improve type safety and overall developer experience.
- **Lifecycle Hook Routing**: Shifted core orchestration of `OnSessionStartHook`, turn-level hooks (`PRE_TURN` and `POST_TURN`), and `OnSessionEndHook` to the connection layer, implementing the Python-side `HookRouter` for event routing.
- **Public API Cleanup**: Hid internal validation methods on media and error classes by prefixing them with an underscore (`_validate_mime_type` and `_from_pydantic` on validation errors).

### Fixed
- **Historical Step Absorption**: Ensured historical step absorption is properly drained during initialization to prevent persistence non-linearity issues in `Conversation`.
- **Python 3.14 Compatibility**: Resolved potential name shadowing in the `Conversation` class by renaming the top-level connection module import.

---

## [0.1.4] - 2026-06-18

This release introduces major architectural refactorings, public API standardizations, and key new capabilities centered on centralizing model configurations to natively support multi-model backends, exposing a new built-in Google Web Search capability, enabling environment variable passing for Model Context Protocol (MCP) servers, and simplifying the agent initialization flow by removing dynamic runtime registrations.

### Added
- **Built-in Web Search Tool**: Exposes the `SEARCH_WEB` tool directly within the SDK, enabling agents to leverage Google Search for grounded real-time information retrieval, complete with new developer examples (`web_tools.py`).
- **MCP Server Environment Variables**: Added support for configuring and passing custom environment variables to launched stdio servers via the new `env` field in `McpStdioServer`.
- **Base URL and HTTP Headers Support**: Out-of-the-box support for setting custom base URLs and HTTP headers.
- **Image Generation Aspect Ratio**: Updated the SDK model config and wrapper to support specifying `aspect_ratio` within the image creation tool configuration.

### Changed
- **Centralized Multi-Model Configuration**: Replaces legacy singular `gemini_config` options with a unified, repeated `models` collection on `AgentConfig` and `LocalAgentConfig` to support multi-model routing, fallback strategies, and automated selection helpers.
- **Agent Session Lifecycle & API Standardization**: Improves runtime safety by removing dynamic post-initialization hook and trigger registration in favor of session creation-time declarations.
- **Top-Level Package Exports**: Exposed core SDK constructs (including `Content`, `Image`, `Document`, `Audio`, `Video`, `from_file`, `BuiltinTools`, and `SystemInstructions`) directly under the `google.antigravity` root module for easier access.
- **Hook Base Class Exports**: Consolidated the base hooks implementation by exporting `DecideHook`, `InspectHook`, and `TransformHook` from the hooks package root.
- **Top-Level Policy Package**: Created a new top-level policy package to clean up hook and workspace path validation dependencies.
- **Relocated Trigger Types**: Moved the `FileChange` model and `FileChangeKind` enum from `types.py` to the specialized triggers package.

---

## [0.1.3] - 2026-06-11

This release introduces per-server MCP timeout configurations and improves local connection error handling.

### Added
- **Per-Server MCP Timeout**: Added configuration support to set custom timeouts (in seconds) for individual MCP servers (`BaseMcpServerConfig.timeout_seconds`).
- **Terminal Error Propagation**: The local connection now propagates terminal trajectory errors from the `localharness` binary as structured `AntigravityExecutionError` exceptions in the Python SDK during step collection.

---

## [0.1.2] - 2026-06-04

This release adds Windows platform support, introduces programmatic turn-level stream cancellation, simplifies safety policy configurations for the Model Context Protocol (MCP), and removes the deprecated MCP SSE transport.

### Added
- **Windows Platform Support**: Native compatibility added for Windows x86_64 and ARM64 environments. Path and file URI resolution now correctly handles drive letters and directory separators under Windows.
- **Programmatic Turn-Level Cancellation**: Added programmatic stream cancellation via `ChatResponse.cancel()`. This programmatically aborts active generation turns directly from the client and raises `AntigravityCancelledError` (subclass of `asyncio.CancelledError`) to cleanly signal cancellation in the async flow (`examples/getting_started/cancellation.py`).

### Changed
- **Direct MCP Safety Policy Configuration**: Overloaded `policy.allow`, `policy.deny`, and `policy.ask_user` to accept server configurations (`BaseMcpServerConfig`) directly instead of typing namespaced string paths. Policy evaluation follows a 9-level precedence model (Specific > Prefix Wildcard > Global Wildcard) with longest-match prefix validation to protect against collisions.

### Removed
- **Deprecation of SSE Transport**: Removed the legacy `McpSseServer` configuration and connection handlers in favor of standard Stdio and Streamable HTTP connection strategies.

---

## [0.1.1] - 2026-05-29

This release focuses on significant enhancements to the Model Context Protocol (MCP) integration, adds native Vertex AI authentication support, improves robustness with better error handling, and includes critical fixes for token usage tracking.

### Added
- **MCP Tool Filtering & Simplified Policies**: Added support for `enabled_tools` (allowlist) and `disabled_tools` (denylist) in server configurations, and overloaded safety policy helpers (`policy.allow`, `policy.deny`, `policy.ask_user`) to accept the MCP server configuration object directly.
- **Vertex AI Authentication**: Integrated native support for Vertex AI authentication in the Python SDK.

### Changed
- **MCP Tool Prefixing & Validation**: The SDK now automatically namespaces and prefixes MCP tools (`mcp_{server_name}_{tool_name}`) to prevent name collisions when connecting multiple MCP servers. The `name` field is now mandatory in MCP server configurations and validated as a proper Python identifier.
- **Improved Error Handling**: The SDK now raises explicit, descriptive exceptions for terminal errors rather than failing silently.

### Fixed
- **Structured Output Token Tracking**: Fixed a bug where token `usage_metadata` was not correctly returned when structured output (`response.structured_output()`) was requested.
- **Type Checking Warnings**: Fixed `pytype` warnings (including `wrong-keyword-args` in `LocalAgentConfig`) across several modules.
