# Google Antigravity SDK Connections

The `connections` package provides the abstraction layer for interacting with
agent backends. It decouples the higher-level SDK components (like
`Conversation` and `Agent`) from the specific transport and process
management details of where the agent is running.

## Core Abstractions

### `Connection`

`Connection` is an abstract base class that represents a live session with an
agent backend. Layer 2 APIs depend ONLY on this interface.

Key methods and properties:
- `send(prompt, **kwargs)`: Sends a prompt to the agent.
- `receive_steps()`: An async iterator that yields `Step` objects as they are
  completed by the agent.
- `disconnect()`: Disconnects the session and releases resources.
- `cancel()`: Cancels the current turn.
- `is_idle`: Property indicating if the connection is idle.
- `conversation_id`: Property returning the conversation identifier.

### `ConnectionStrategy`

`ConnectionStrategy` is an abstract base class for establishing a `Connection`.
It handles process management, transport setup, authentication, and health
checking specific to a backend type.

Key methods:
- `connect()`: Returns the established `Connection`.
- `__aenter__()` and `__aexit__()`: Support for use as an async context manager
  to manage the backend lifecycle.

### `AgentConfig`

`AgentConfig` is an abstract base class for agent configuration. Each
`ConnectionStrategy` defines a concrete subclass with the config fields it needs
(e.g., `LocalAgentConfig` for `LocalConnectionStrategy`).

Key methods:

- `create_strategy(tool_runner, hook_runner)`: Abstract method that creates the
  `ConnectionStrategy` for this config.

## Package-per-Strategy Convention

Each connection strategy lives in its own sub-package under `connections/`.
A strategy's sub-package co-locates its implementation, config, proto bindings,
and tests:

```
connections/
├── connection.py            # ABCs: Connection, ConnectionStrategy, AgentConfig
├── connection_test.py
├── README.md
└── local/        # LocalConnection strategy
    ├── __init__.py           # Re-exports public API
    ├── local_connection.py   # LocalConnection, LocalConnectionStrategy
    ├── local_connection_config.py  # LocalAgentConfig
    ├── local_connection_test.py
    └── localharness_pb2.py   # Proto bindings for the local harness
```

When adding a new connection strategy, create a new sub-package following this
pattern. For example, a remote connection strategy would go in
`connections/remote_connection/`.

## Implementations

### `LocalConnection` (`local/`)

`LocalConnection` (and its corresponding `LocalConnectionStrategy`) connects to
a Go-based local harness.

- **Transport**: It uses WebSockets to communicate with the Go harness.
- **Protocol**: It communicates using protobuf messages (`OutputEvent`,
  `InputEvent`, `StepUpdate`, etc.) serialized to JSON.
- **Config**: `LocalAgentConfig` in `local_connection_config.py`.
- **Features**:
    - Handles tool calls by executing them via `ToolRunner` and sending results
      back.
    - Handles question requests from the harness and dispatches them via
      `HookRunner` (interaction hooks).
    - Dispatches session start/end and turn hooks.
    - Supports loading skills from specified paths via `skills_paths`.
    - Supports overriding the application data directory for generated artifacts and media via `app_data_dir`.
    - Supports configuring model API and output retry behavior with exponential backoff via `retry_config`.

## Usage Example

```python
from google.antigravity.connections.local import LocalConnectionStrategy
from google.antigravity.types import GeminiAPIEndpoint, ModelConfig, ModelType

strategy = LocalConnectionStrategy(
    binary_path="/path/to/localharness",
    models=[
        ModelConfig(
            name="gemini-3.8-flash",
            types=[ModelType.TEXT],
            endpoint=GeminiAPIEndpoint(api_key="..."),
        )
    ],
    skills_paths=["/path/to/skills"],
)

async with strategy:
    connection = strategy.connect()
    await connection.send("Hello")
    async for step in connection.receive_steps():
        print(step)
```

## Files

- `connection.py`: Defines `Connection`, `ConnectionStrategy`, and
  `AgentConfig` interfaces.
- `local/`: Local harness connection strategy package.
  - `local_connection.py`: Implements `LocalConnection` and
    `LocalConnectionStrategy`.
  - `local_connection_config.py`: Implements `LocalAgentConfig`.
  - `localharness_pb2.py`: Generated protobuf bindings.
