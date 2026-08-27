# Google Antigravity SDK

The Google Antigravity SDK is a Python SDK for building AI agents powered by
Antigravity and Gemini. It provides a secure, scalable, and stateful
infrastructure layer that abstracts the agentic loop, letting you focus on what
your agent *does* rather than how it runs.

## Installation

```sh
pip install google-antigravity
```

> [!IMPORTANT]
> The Google Antigravity SDK relies on a compiled runtime binary that is
> included in the platform-specific wheels published to
> [PyPI](https://pypi.org/project/google-antigravity/). **Cloning this
> repository alone is not sufficient to run the SDK.** Always install from
> PyPI with `pip install google-antigravity` to obtain the binary.

## Quickstart

Get started by running one of the [`examples/`](examples/), such as the
`hello_world` example with:

```sh
export GEMINI_API_KEY="your_api_key_here"
python ./examples/getting_started/hello_world.py
```
## Gemini Enterprise Agent Platform (formerly Vertex AI)

To use the SDK with Gemini Enterprise Agent Platform (formerly Vertex AI),
the SDK supports two authentication modes:

### 1. Express Mode (API Key)

For fast setup without requiring Google Cloud projects, regional configuration,
or Application Default Credentials (ADC), provide an API key with `vertex=True`:

```python
from google.antigravity import Agent, LocalAgentConfig

config = LocalAgentConfig(
    vertex=True,
    api_key="your_api_key_here",
)

async with Agent(config) as agent:
    response = await agent.chat("Hello!")
    print(await response.text())
```

### 2. Standard Mode (Project & Location with ADC)

For enterprise deployments routing to regional endpoints, configure
`LocalAgentConfig` with `vertex=True`, `project`, and `location`. By default,
this mode authenticates via Application Default Credentials (ADC).

```python
from google.antigravity import Agent, LocalAgentConfig

config = LocalAgentConfig(
    vertex=True,
    project="your-gcp-project",
    location="us-central1",
)

async with Agent(config) as agent:
    response = await agent.chat("Hello!")
    print(await response.text())
```

Alternatively, you can leave these fields unset in `LocalAgentConfig` and
export the environment variables instead:

```sh
# Either GOOGLE_GENAI_USE_VERTEXAI or GOOGLE_GENAI_USE_ENTERPRISE enable Vertex.
export GOOGLE_GENAI_USE_VERTEXAI=True
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

Explicit kwargs always take precedence over env vars.

Ensure you have authenticated locally before running the agent in Standard Mode:

```sh
gcloud auth application-default login
```

See [vertex.py](./examples/getting_started/vertex.py) for a complete example.

## Concepts

### Simple Agent

The `Agent` class is the easiest way to get started. It manages the full
lifecycle — binary discovery, tool wiring, hook registration, and policy
defaults — behind a single async context manager.

The `system_instructions` parameter is optional.

```python
import asyncio
from google.antigravity import Agent, LocalAgentConfig

async def main():
    config = LocalAgentConfig(
        system_instructions="You are an expert assistant for codebase navigation.",
        # api_key="your_api_key_here",
    )
    async with Agent(config) as agent:
        response = await agent.chat("What files are in the current directory?")
        print(await response.text())

async def run():
    await main()

if __name__ == "__main__":
    asyncio.run(run())
```

### Streaming Responses

To stream agent output in real-time (e.g., for fluid UI or console applications), simply iterate over the `ChatResponse` object using an `async for` loop. The stream wrapper natively yields conversational `str` text tokens as they arrive, with zero network overhead:

```python
import asyncio
import sys
from google.antigravity import Agent, LocalAgentConfig

async def main():
    config = LocalAgentConfig()
    async with Agent(config) as agent:
        # Returns instantly — does not block
        response = await agent.chat("Write a short poem about space.")
        
        async for token in response:
            sys.stdout.write(token)
            sys.stdout.flush()
        print()

asyncio.run(main())
```

### Sugared Thoughts & Tool Call Streams (Advanced)

For more complex use cases, you can also stream internal model reasoning/thinking or intercept tool call dispatches in real-time using dedicated async stream properties:

```python
# 1. Stream reasoning/thinking deltas
async for thought in response.thoughts:
    show_thinking_bubble(thought)

# 2. Stream strongly-typed ToolCall events
async for call in response.tool_calls:
    show_executing_spinner(call.name)
```

By default, `Agent` runs in **read-only mode** for safety. Pass
`capabilities=CapabilitiesConfig()` to enable all tools (including writes).

### Interactive Loop

```python
from google.antigravity import LocalAgentConfig, CapabilitiesConfig
from google.antigravity.utils.interactive import run_interactive_loop

config = LocalAgentConfig(
    # api_key="your_api_key_here",
    capabilities=CapabilitiesConfig(),
)
await run_interactive_loop(config)
```

### Advanced Usage with Conversation

For full control over the connection lifecycle, use `Conversation` with a
`ConnectionStrategy` directly. `Conversation` is a stateful session that
accumulates step history, provides a `chat()` convenience method, and exposes
state introspection:

```python
import asyncio
from google.antigravity.connections.local import LocalConnectionStrategy
from google.antigravity.conversation.conversation import Conversation
from google.antigravity.tools.tool_runner import ToolRunner

async def main():
    tool_runner = ToolRunner()
    strategy = LocalConnectionStrategy(
        tool_runner=tool_runner,
    )
    
    async with Conversation.create(strategy) as conversation:
        # High-level: one-call send + collect
        response = await conversation.chat("What files are here?")
        print(await response.text())
        
        # Step history accumulates automatically
        print(f"Total steps: {len(conversation.history)}")
        print(f"Turns: {conversation.turn_count}")
        print(f"Last response: {conversation.last_response}")
        
        # Low-level: streaming steps
        await conversation.send("Tell me more.")
        async for step in conversation.receive_steps():
            if step.is_complete_response:
                print(step.content)

asyncio.run(main())
```

## Features

### Multimodal Ingestion

Pass rich multimedia file attachments (images, videos, audio, and documents) to the agent alongside textual instruction prompt lists.

You can attach assets **directly using content classes** (perfect for in-memory bytes) or **conveniently from a filesystem path** (which automatically resolves types and guesses MIME formats):

```python
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.types import Image, from_file

config = LocalAgentConfig(system_instructions="You are an expert software architect.")
async with Agent(config) as agent:
    # 1. Flat filesystem shortcut (automatically resolves as types.Document)
    pdf_spec = from_file("spec.pdf")
    
    # 2. Direct constructor instantiation (perfect for in-memory raw bytes)
    chart_image = Image(
        data=b"raw_png_bytes_here", 
        mime_type="image/png", 
        description="Architecture blueprint"
    )
    
    # Send a mixed list of text instructions and content classes
    prompt = [
        "Analyze this chart against the specification and list three security vulnerabilities:",
        chart_image,
        pdf_spec
    ]
    response = await agent.chat(prompt)
    print(await response.text())
```

### Custom Tools

Register Python functions as tools that the agent can call:

```python
def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"It's sunny in {city}."

config = LocalAgentConfig(
    tools=[get_weather],
)
async with Agent(config) as agent:
    response = await agent.chat("What's the weather in Tokyo?")
```

### MCP Integration

Connect to external [MCP](https://modelcontextprotocol.io/) servers and expose
their tools to the agent:

```python
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.types import McpStdioServer

config = LocalAgentConfig(
    mcp_servers=[McpStdioServer(name="my_server", command="npx", args=["my-mcp-server"])],
)
async with Agent(config) as agent:
    response = await agent.chat("Use the MCP tools to help me.")
```

### Hooks and Policies

Control agent behavior with a declarative policy system:

```python
from google.antigravity import LocalAgentConfig, CapabilitiesConfig
from google.antigravity.hooks.policy import deny, allow, ask_user, enforce
from google.antigravity.utils.interactive import run_interactive_loop

policies = [
    deny("*"),                          # Block all tools by default
    allow("view_file"),                 # Allow reading files
    ask_user("run_command", handler=my_handler),  # Ask before running commands
]

config = LocalAgentConfig(
    capabilities=CapabilitiesConfig(),
    policies=policies,
)
await run_interactive_loop(config)
```

### Triggers

Run background tasks that react to external events and push messages into the
agent:

```python
from google.antigravity import LocalAgentConfig
from google.antigravity.triggers import every
from google.antigravity.utils.interactive import run_interactive_loop

async def check_status(ctx):
    await ctx.send("Check the deployment status.")

config = LocalAgentConfig(
    triggers=[every(60, check_status)],
)
await run_interactive_loop(config)
```

## Architecture

The SDK follows a three-layer architecture:

| Layer | Purpose | Key Classes |
|:------|:--------|:------------|
| **Layer 1** — Simplified | High-level, batteries-included entry point | `Agent` |
| **Layer 2** — Session | Stateful session with history and convenience methods | `Conversation`, `ChatResponse`, `Step`, `ToolCall`, `AgentConfig`, `HookRunner`, `ToolRunner`, `TriggerRunner` |
| **Layer 3** — Adapter | Transport and backend abstraction | `Connection`, `ConnectionStrategy`, `LocalConnection` |

## Component Documentation

For more detailed documentation on specific components, see:

-   [Agent](google/antigravity/agent.py) — High-level, batteries-included entry point.
-   [Connections](google/antigravity/connections/README.md) — Transport and backend abstraction.
-   [Conversation](google/antigravity/conversation/README.md) — Stateful session management.
-   [Hooks](google/antigravity/hooks/README.md) — Agent lifecycle interception and policies.
-   [MCP](google/antigravity/mcp/README.md) — Model Context Protocol integration.
-   [Tools](google/antigravity/tools/README.md) — In-process tool execution.
-   [Triggers](google/antigravity/triggers/README.md) — Background tasks and external events.

## License

[Apache License 2.0](LICENSE)
