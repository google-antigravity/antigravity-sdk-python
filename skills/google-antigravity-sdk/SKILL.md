---
name: google-antigravity-sdk
description: "Design, implement, and debug autonomous AI agents and multi-agent systems using the Google Antigravity (AGY) SDK. ACTIVATE this skill when the user wants to create, configure, or orchestrate Google Antigravity agents."
---

# Google Antigravity SDK

## Installation & Setup

Before proceeding with any Google Antigravity tasks, ensure the environment is
ready:

-   **Verify Applicability**: If operating in an existing codebase, verify that
    using this Python SDK is possible and appropriate for the project.
-   **Check Dependencies**: Check if `google-antigravity` is listed in the
    project's dependencies (e.g., `requirements.txt`, `pyproject.toml`).
-   **Install Package**: Ensure the `google-antigravity` Python package is
    installed.
-   **Authentication Setup**: Check for a valid `GEMINI_API_KEY` environment
    variable or a `.env` file (required to access Gemini models).
    -   If credentials are missing, you MUST actively help the user get set up
        with an API key by providing the following link:
        -   Default to Google AI Studio:
            `https://aistudio.google.com/app/api-keys`
    -   Explain that the API key can be passed explicitly in code as shorthand
        (e.g., `LocalAgentConfig(api_key="...")`) or automatically read from the
        environment.
    -   For Gemini Enterprise Agent Platform (formerly Vertex AI)
        authentication, the SDK supports both Standard Mode and Express Mode:
        -   **Standard Mode (ADC)**: Instruct the user to run
            `gcloud auth application-default login` and configure the agent with
            `vertex=True` along with `project` and `location` in
            `LocalAgentConfig`.
        -   **Express Mode (API Key)**: Configure the agent with `vertex=True`
            along with `api_key="your-express-api-key"` in `LocalAgentConfig`
            (no ADC or regional project/location needed).
    -   **Note**: For local models (`LiteRTAgentConfig` or
        `LocalOpenAIAgentConfig`), no API key or cloud credentials are needed.
        See `references/local_models.md` for setup details.

## Routing Table

Use the following information to dig deeper into specific topics based on the
user request. Read the referenced files or explore the directories to find
relevant information.

### References

-   If the user needs to understand the high-level overview and core concepts of
    the Google Antigravity SDK (Agent, Conversation, Connection), read
    `references/architecture.md`.
-   If the user needs to perform advanced agent configuration (e.g., selecting
    appropriate models, configuring execution behavior via `agent_behavior`—defaulting
    to autonomous vs interactive—or configuring connection reliability), or
    understand the critical rules for model identifiers to avoid assumptions,
    read `references/agent_configuration.md`.
-   If the user needs to extend an agent's capabilities by integrating Model
    Context Protocol (MCP) servers, or configure tool permissions for the agent,
    read `references/mcp_integration.md`.
-   If the user needs to define safety policies, resolve execution order,
    restrict agent actions using predicates, or run terminal commands inside an
    OS-level sandbox, read `references/safety_policies.md`.
-   If the user needs to debug failed agents, stream logs, or implement error
    recovery using hooks to make agents robust, read
    `references/error_handling.md`.
-   If the user needs to monitor costs, track token usage (including thinking
    tokens), or build custom audit logs for advanced monitoring, read
    `references/observability.md`.
-   If the user needs to see a list of built-in tools and understand their default state, read `references/built_in_tools.md`.
-   If the user needs to run agents locally using on-device models (e.g., Gemma
    via LiteRT, or via OpenAI-compatible APIs), understand hardware
    requirements, or set up a local model environment, read
    `references/local_models.md`.

### Examples

-   If the user needs to implement basic agent behavior, streaming responses, or
    expose internal thoughts, read `examples/getting_started/hello_world.md`.
-   If the user needs to customize or override default retry behavior and
    exponential backoff for API errors or schema validation, read
    `examples/getting_started/customizing_retries.md`.
-   If the user needs to equip an agent with custom capabilities (tools) derived
    from Python functions, or maintain agent state across tool execution, read
    `examples/getting_started/custom_tool.md`.
-   If the user needs to shape an agent's persona, define its system
    instructions, or dynamically adapt its behavior, read
    `examples/getting_started/persona_config.md`.
-   If the user needs to build multimodal agents capable of processing images
    and PDFs, or generating visual content, read
    `examples/getting_started/multimodal.md`.
-   If the user needs to implement multi-agent delegation, allowing a main agent
    to spawn and orchestrate subagents, or configure multi-tier nested subagent
    hierarchies (using `max_subagent_depth` and `allowed_subagents`), read
    `examples/getting_started/subagents.md`.
-   If the user needs to connect an agent to external services via MCP (Stdio or
    SSE), read `examples/getting_started/mcp_tools.md`.
-   If the user needs to create proactive agents that respond to time-based
    events or file system triggers in the background, read
    `examples/getting_started/periodic_trigger.md`.
-   If the user needs to intercept agent lifecycle events (e.g., pre/post turn,
    stop, tool execution, errors) to customize execution flow, read
    `examples/getting_started/hooks.md`.
-   If the user needs to implement turn-level cancellation or programmatic
    stream aborts, read `examples/getting_started/cancellation.md`.
-   If the user needs to implement persistent agents that remember past
    interactions across sessions, read
    `examples/getting_started/persistence.md`.
-   If the user needs to override the default application data directory
    for agent artifacts, scratch files, and media storage, read
    `examples/getting_started/app_data_dir_override.md`.
-   If the user needs an agent to output structured data (e.g., JSON matching a
    Pydantic schema) for reliable integration, read
    `examples/getting_started/structured_output.md`.
-   If the user needs to add, configure, or load agent skills into the Google
    Antigravity SDK agent, read `examples/getting_started/agent_skills.md`.
-   If the user needs to enable and use built-in web tools (like Google Search
    or URL fetching) with the agent, read
    `examples/getting_started/web_tools.md`. (Note: when fetching massive web
    pages or articles, pair `read_url_content` with `view_file` to inspect
    cached disk files).
-   If the user needs to enforce session operational limits (model or
    tool calls) or proactive token budget controls (input, output, or
    total tokens) and handle `StopReason`, read
    `examples/getting_started/budget_limits.md`.
-   If the user needs to set up and run a local model agent (LiteRT with Gemma,
    or an OpenAI-compatible server like Ollama), including model download,
    hardware requirements, and context window configuration, read
    `examples/getting_started/local_models.md`.
-   If the user needs to configure conversation context limits or background
    trajectory checkpointing (compaction cadence) to handle long-running
    sessions, read `examples/getting_started/compaction.md`.

