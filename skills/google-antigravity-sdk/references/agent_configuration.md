


# Advanced Agent Configuration Guide

This guide provides instructions on how to perform advanced configuration for
Google Antigravity SDK agents.

## Model Selection

### Default Model

Google Antigravity SDK's default model is `gemini-3.8-flash`.

### Default Image Generation Model

Google Antigravity SDK's default image generation model is `gemini-3.1-flash-lite-image`.

### Finding Valid Models

To find the most up-to-date list of valid Gemini model identifiers, refer to the
official documentation: -
[Google AI Studio Documentation](https://ai.google.dev/gemini-api/docs/models/gemini)

## CRITICAL RULE: Never Assume Valid Model Identifiers

> [!IMPORTANT] **Do not assume valid model identifiers.** Avoid guessing model
> names or assuming they follow a specific pattern. Always verify the valid
> identifiers from official documentation or user context before using them.

> [!IMPORTANT] **Avoid setting the model explicitly unless requested.** It is
> generally better to leave the model unset to use the default behavior, unless
> the user has explicitly requested a specific model.

## Advanced Configuration Examples

Here are small code snippets demonstrating advanced configurations using
`LocalAgentConfig`.

### Basic Configuration with Model Selection

```python
from google.antigravity import Agent, LocalAgentConfig

config = LocalAgentConfig(
    model="gemini-3.8-flash",
)
async with Agent(config=config) as agent:
    # Use the agent
    pass
```

### Agent Execution Behavior (`agent_behavior`)

The SDK supports two operational execution behaviors via `types.AgentBehavior`:

-   `AgentBehavior.AUTONOMOUS` (**default**): Non-interactive, automated execution.
    The agent is incentivized to accomplish the task on its own from start to
    finish.
-   `AgentBehavior.INTERACTIVE`: Collaborative execution with a human. The agent
    asks clarifying questions (via `BuiltinTools.ASK_QUESTION`), enables
    interactive planning, and keeps the user in the loop.

Configure `agent_behavior` via `CapabilitiesConfig`:

```python
from google.antigravity import Agent, LocalAgentConfig, types

config = LocalAgentConfig(
    capabilities=types.CapabilitiesConfig(
        agent_behavior=types.AgentBehavior.INTERACTIVE,
    ),
)
async with Agent(config=config) as agent:
    # Agent will operate with interactive behavior, asking questions if needed
    pass
```

> [!NOTE]
> `agent_behavior` defaults to `AgentBehavior.AUTONOMOUS`. If you enable interactive tools such as `BuiltinTools.ASK_QUESTION` without setting `agent_behavior=AgentBehavior.INTERACTIVE`, a validation warning will be logged.

### Nested Subagents & Depth Controls (`max_subagent_depth`, `allowed_subagents`)

The SDK supports multi-tier hierarchical subagent execution:

-   `max_subagent_depth`: Configures the session-wide subagent recursion depth
    ceiling on `CapabilitiesConfig` (root conversation is depth 0).
-   `allowed_subagents`: An explicit allowlist of subagent names that the root
    agent (on `CapabilitiesConfig`) or a specific subagent (on
    `SubagentCapabilities`) is permitted to invoke.

```python
from google.antigravity import Agent, LocalAgentConfig, types

# Configure a subagent that can invoke other subagents
researcher = types.SubagentConfig(
    name="researcher",
    description="Research agent with subagent delegation capability",
    capabilities=types.SubagentCapabilities(
        enabled_tools=[
            types.BuiltinTools.VIEW_FILE,
            types.BuiltinTools.START_SUBAGENT,
        ],
        allowed_subagents=["fact_checker"],
    ),
)

# Root agent configured with a max depth of 3
config = LocalAgentConfig(
    subagents=[researcher],
    capabilities=types.CapabilitiesConfig(
        enable_subagents=True,
        max_subagent_depth=3,
        allowed_subagents=["researcher"],
    ),
)
```

### Gemini Enterprise Agent Platform (formerly Vertex AI) Configuration

To configure the agent to use Gemini Enterprise Agent Platform (formerly Vertex
AI) instead of Gemini Developer API:

```python
from google.antigravity import Agent, LocalAgentConfig

# 1. Express Mode (API Key) - authenticates against aiplatform.googleapis.com
express_config = LocalAgentConfig(
    vertex=True,
    api_key="your-express-api-key",
)

# 2. Standard Mode (ADC) - regional routing with project and location
standard_config = LocalAgentConfig(
    vertex=True,
    project="your-gcp-project",
    location="us-central1",
)

async with Agent(config=express_config) as agent:
  # Use the agent with Gemini Enterprise Agent Platform
  pass
```

Note: In Standard Mode, Gemini Enterprise Agent Platform authentication relies on
Application Default Credentials (ADC); ensure you have run
`gcloud auth application-default login` in your environment. In Express Mode,
only the `api_key` and `vertex=True` are required.

### Prioritized Inference (`service_tier="priority"`)

To enable Gemini Prioritized Inference, manually construct a `GeminiAPIEndpoint`
or `VertexEndpoint` with `GeminiModelOptions`:

```python
from google.antigravity import (
    Agent,
    GeminiAPIEndpoint,
    LocalAgentConfig,
    types,
)

# Configure priority inference by manually constructing an endpoint.
options = types.GeminiModelOptions(service_tier=types.ServiceTier.PRIORITY)
endpoint = GeminiAPIEndpoint(options=options)
config = LocalAgentConfig(endpoint=endpoint)

async with Agent(config=config) as agent:
  response = await agent.chat("Explain quantum computing in one sentence.")

  # Inspect usage metadata to detect server-side rate limit downgrades.
  if (
      response.usage_metadata
      and response.usage_metadata.service_tier == types.ServiceTier.STANDARD
  ):
    print("Notice: Request was gracefully downgraded to standard tier.")
```

> [!IMPORTANT] **Pricing Notice:** Priority tier requests are billed at a higher
> rate than Standard tier requests. When overflow traffic is gracefully
> downgraded to Standard tier due to dynamic rate limiting, those downgraded
> requests are billed at standard rates. Please check the linked documentation
> for specific pricing, fallback thresholds, and feature details:
> [Gemini Priority Inference Documentation](https://ai.google.dev/gemini-api/docs/priority-inference).

### Application Data Directory Override (Artifact & Scratch Storage)

By default, the agent stores generated artifacts (like `task.md`), scratch
files, and uploaded media under `~/.gemini/antigravity/brain/`. You can override
this location by specifying an absolute path in `app_data_dir`:

```python
from google.antigravity import Agent, LocalAgentConfig

config = LocalAgentConfig(
    app_data_dir="/absolute/path/to/custom/storage",
)
async with Agent(config=config) as agent:
    # Generated files and artifacts will be written inside the custom directory
    pass
```

> [!IMPORTANT] **The path must be an absolute path.** Passing relative paths or
> unexpanded tildes (`~/`) will trigger a validation error.

### System Instructions and Personas

You can configure system instructions directly in the `LocalAgentConfig`:

```python
config = LocalAgentConfig(
    system_instructions="You are an expert software architect.",
)
```

For a more detailed guide and complex persona examples, see
[persona_config.md](../../examples/getting_started/persona_config.md).

### Custom Tools

You can add custom tools to your agent:

```python
from google.antigravity import Agent, LocalAgentConfig

config = LocalAgentConfig(
    tools=[my_custom_tool_function],
)
```

For a full guide on creating and using custom tools, see
[custom_tool.md](../../examples/getting_started/custom_tool.md).

### MCP Integration

To configure Model Context Protocol (MCP) servers:

```python
from google.antigravity import Agent, LocalAgentConfig, types

config = LocalAgentConfig(
    mcp_servers=[
        types.McpStreamableHttpServer(
            name="my_mcp_server",
            url="http://localhost:8080",
        )
    ],
)
```

For more details, see [mcp_integration.md](mcp_integration.md).

### Local Model Configuration

The SDK supports running agents entirely on-device without an API key. Two
additional config classes are available:

-   `LiteRTAgentConfig`: For running Gemma models locally via LiteRT-LM.
-   `LocalOpenAIAgentConfig`: For connecting to any OpenAI-compatible local
    server (e.g., Ollama, LM Studio).

Both config classes support the `.lightweight()` method (e.g.,
`LiteRTAgentConfig(...).lightweight()`), which automatically configures core
development tools, prunes system instructions for smaller context windows,
disables subagents, and tunes context compaction.

For full setup instructions, hardware requirements, and configuration details,
see [local_models.md](local_models.md).

### Custom Environment Variables (Subprocess & Shell Isolation)

You can pass a custom dictionary of environment variables using `env` in `LocalAgentConfig`. These variables override any variables with the same name in the parent process's environment when launching `localharness` and are inherited by shell tool execution (`run_command`):

```python
from google.antigravity import Agent, LocalAgentConfig
import os

config = LocalAgentConfig(
    env={"PATH": "/custom/bin:" + os.environ.get("PATH", ""), "MY_CUSTOM_VAR": "foo"},
)
```

### `run_command` Configuration (`RunCommandConfig`)

Configure the built-in `run_command` tool via `RunCommandConfig`, including
running commands inside an OS-level sandbox with `enable_sandbox`:

```python
from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.hooks import policy

config = LocalAgentConfig(
    capabilities=types.CapabilitiesConfig(
        run_command_config=types.RunCommandConfig(enable_sandbox=True),
    ),
    policies=[policy.allow_all()],
)
```

For details and caveats, see
[safety_policies.md](safety_policies.md#defense-in-depth-os-level-command-sandboxing).

### Session Budget Controls & Stop Reasons

You can configure session operational limits (`max_model_calls`, `max_tool_calls`) and proactive token budget controls (`max_input_tokens`, `max_output_tokens`, `max_total_tokens`) using `BudgetConfig`:

```python
from google.antigravity import Agent, LocalAgentConfig, types

config = LocalAgentConfig(
    budget_config=types.BudgetConfig(
        max_model_calls=10,
        max_tool_calls=20,
        max_total_tokens=100_000,
    ),
)
```

For a full guide and multi-turn stop reason handling examples, see [budget_limits.md](../../examples/getting_started/budget_limits.md).

### Context Compaction & Token Limits (`compaction_config`)

Antigravity manages conversation context using a two-stage sliding-window pipeline:

1. **Background Checkpointing**: A background model pre-computes cumulative trajectory summaries (checkpoints) at regular token intervals (`checkpoint_interval_tokens`). Checkpoint generation runs asynchronously and silently in parallel without modifying or truncating the active prompt.
2. **Prompt Eviction (Compaction)**: When cumulative prompt tokens reach the context ceiling (`max_context_tokens`), the prompt snaps back to the latest completed checkpoint. Earlier checkpoints and older turns preceding the latest checkpoint are evicted from the prompt, while recent turns between the latest checkpoint and the current turn are preserved verbatim with full fidelity.

You can configure both dials using `CompactionConfig`:

```python
from google.antigravity import Agent, LocalAgentConfig, types

config = LocalAgentConfig(
    compaction_config=types.CompactionConfig(
        checkpoint_interval_tokens=40_000,
        max_context_tokens=100_000,
    ),
)
```

> [!NOTE]
> When `compaction_config` is omitted (or fields are left unset), the backend's default cadence and context ceiling are used. If configuring custom values, `checkpoint_interval_tokens` cannot exceed `max_context_tokens`.

For a full guide and code examples, see [compaction.md](../../examples/getting_started/compaction.md).

