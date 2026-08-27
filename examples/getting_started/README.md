# Getting Started with Google Antigravity SDK

This directory contains minimal, single-file examples demonstrating the core features of the Google Antigravity SDK. 

## 🚀 Quickstart

To get started immediately, install the SDK, set your API key, and run the basic chat snippet.

### 1. Install & Authenticate

```bash
pip install google-antigravity
export GEMINI_API_KEY="your_api_key_here"
```
*(Note: Alternatively, you can pass your key explicitly in code via `LocalAgentConfig(api_key="...")`).*

### 2. Execute Your First Turn

Create a file named `quickstart.py` and run it:

```python
import asyncio
from google.antigravity import Agent, LocalAgentConfig

async def main():
    # Initialize the agent configuration. It automatically picks up GEMINI_API_KEY from the environment.
    config = LocalAgentConfig()
    async with Agent(config) as agent:
        response = await agent.chat("Explain quantum computing in one sentence.")
        print(await response.text())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🗂️ Examples Index

Once you have the quickstart running, explore the modular examples below to understand the SDK's capabilities. Run any example directly from your terminal (e.g., `python3 hello_world.py`).

### Core Foundations
The essential building blocks for initializing, configuring, and prompting agents.
* [hello_world.py](hello_world.py): Initializing an agent, context manager usage, and explicit model configuration.
* [vertex.py](vertex.py): Connecting to Vertex AI (Gemini Enterprise Agent Platform) via Express Mode (API key) or Standard Mode (GCP project & location with ADC).
* [streaming.py](streaming.py): Real-time token streaming and inspecting model reasoning via `response.thoughts`.
* [persona_config.py](persona_config.py): Structuring system instructions and shaping agent identity using `TemplatedSystemInstructions`.
* [prioritized_inference.py](prioritized_inference.py): Configuring Gemini Prioritized Inference (`service_tier="priority"`) and monitoring server-side rate limit downgrades.

### 🛡️ Safety & Governance
Securing agent actions, enforcing operational/token budgets, and keeping
humans in control before executing external tools.

* [budget_limits.py](budget_limits.py): Enforcing session operational limits
  (model calls, tool calls) and proactive token budget controls (input, output,
  and total tokens) with stop reason inspection.
* [policies.py](policies.py): Implementing robust safety policies ("Deny by Default", allowlisting, and `ask_user`).
* [human_in_the_loop.py](human_in_the_loop.py): Interactively pausing execution to request human confirmation or input.

### 🧩 Structured & Multimodal Interactivity
Handling complex inputs and enforcing strict data outputs.
* [multimodal.py](multimodal.py): Processing images/PDFs and generating visual assets.
* [structured_output.py](structured_output.py): Enforcing strictly typed JSON responses matching Pydantic schemas (`response_schema`).

### 🛠️ Tools, Skills, & Delegation
Extending agent capabilities and orchestrating multi-agent workflows.
* [custom_tools.py](custom_tools.py): Defining stateful Python functions as tools using `ToolContext`.
* [agent_skills.py](agent_skills.py): Discovering and loading domain-specific skills from the filesystem (`SKILL.md`).
* [mcp_tools.py](mcp_tools.py): Connecting to external toolsets via the Model Context Protocol (MCP).
* [subagents.py](subagents.py): Spawning and delegating specialized tasks to sub-agents.
* [web_tools.py](web_tools.py): Enabling and utilizing built-in web search and URL content fetching tools.

### ⚙️ Lifecycle, Proactivity, & Observability
Controlling execution flow, reacting to background events, auditing performance, and maintaining session state.
* [hooks.py](hooks.py): Intercepting session and turn lifecycle events (`pre_turn`, `post_turn`).
* [cancellation.py](cancellation.py): Programmatic turn aborts (`response.cancel()`) and clean async task cancellation handling.
* [triggers.py](triggers.py): Running background checks and periodic tasks during active conversations.
* [observability.py](observability.py): Auditing execution, tracking token costs (including thinking tokens), and configuring logging.
* [error_handler.py](error_handler.py): Gracefully recovering from tool execution failures via `@hooks.on_tool_error`.
* [persistence.py](persistence.py): Saving and resuming stateful conversation sessions across restarts using `conversation_id` and `save_dir`.
* [app_data_dir_override.py](app_data_dir_override.py): Overriding the default application data directory for agent artifacts, scratch files, and media storage using `app_data_dir`.
