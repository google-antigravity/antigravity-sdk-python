# Safety Policies in Google Antigravity SDK

Reference guide for configuring access control and safety policies in the Google
Antigravity SDK.

## Overview

The Google Antigravity SDK provides a declarative policy system to control which
tools an agent can execute. Policies are evaluated using a priority-based model
to ensure safety and prevent unauthorized actions.

## Default Behavior

By default, `LocalAgentConfig` uses `policy.confirm_run_command()` which:

-   **Denies** `run_command` (shell execution is blocked)
-   **Allows** all other tools (view, edit, create files, etc.)

This means new agents are **conservative by default** — they cannot execute shell
commands unless you explicitly opt in.

If `workspaces` is set on the config, `policy.workspace_only()` is also
automatically prepended, restricting file tools (`view_file`, `create_file`,
`edit_file`) to the configured workspace directories.

### Interactive Sessions

When using `run_interactive_loop()`, the default deny on `run_command` is
automatically upgraded to `ask_user` — the user gets a y/n confirmation prompt
instead of a hard denial.

### Restoring Permissive Behavior

To allow all tools (including `run_command`), pass `policy.allow_all()`:

```python
from google.antigravity import LocalAgentConfig
from google.antigravity.hooks import policy

config = LocalAgentConfig(
    system_instructions="You are a helpful assistant.",
    policies=[policy.allow_all()],
)
```

## Policy Resolution Order

Policies are evaluated in the following order of precedence (highest to lowest),
supporting 9 levels of priority:

1.  **Specific Deny**: `policy.deny("tool_name", ...)` (e.g.,
    `policy.deny("run_command")` or `policy.deny(server_cfg, ["tool1"])`)
2. **Specific Ask**: `policy.ask_user("tool_name", ...)`
3. **Specific Allow**: `policy.allow("tool_name", ...)`
4.  **Prefix Wildcard Deny**: `policy.deny("server/*", ...)` (e.g.,
    `policy.deny(server_cfg)`)
5.  **Prefix Wildcard Ask**: `policy.ask_user("server/*", ...)`
6.  **Prefix Wildcard Allow**: `policy.allow("server/*", ...)`
7.  **Global Wildcard Deny**: `policy.deny("*", ...)` (e.g.,
    `policy.deny_all()`)
8.  **Global Wildcard Ask**: `policy.ask_user("*", ...)`
9.  **Global Wildcard Allow**: `policy.allow("*", ...)` (e.g.,
    `policy.allow_all()`)

Within each priority group, the **first match wins** (short-circuit evaluation).

## Configuration

Use the `google.antigravity.hooks.policy` module to define policies.

### Allow

Approves tool calls without confirmation. Can be specified as a tool name
string, a wildcard `*`, or using an MCP server config object.

```python
from google.antigravity.hooks import policy

# Allow a standard built-in tool
policy.allow("view_file")

# Allow all tools on an MCP server
policy.allow(mcp_server_cfg)

# Allow a specific subset of tools on an MCP server
policy.allow(mcp_server_cfg, ["tool1", "tool2"])
```

### Deny

Blocks tool calls immediately. Can be specified as a tool name string, a
wildcard `*`, or using an MCP server config object.

```python
from google.antigravity.hooks import policy

# Deny a standard built-in tool
policy.deny("run_command")

# Deny all tools on an MCP server
policy.deny(mcp_server_cfg)

# Deny a specific subset of tools on an MCP server
policy.deny(mcp_server_cfg, ["dangerous_tool"])
```

### Ask User

Requires user confirmation before execution. Must provide a handler. Can be
specified as a tool name string, a wildcard `*`, or using an MCP server config
object.

```python
from google.antigravity.hooks import policy

async def my_approval_handler(tool_call):
  # Custom logic to ask user or auto-approve
  # Return True to allow, False to deny
  return True

# Ask for confirmation on a standard tool
policy.ask_user("run_command", handler=my_approval_handler)

# Ask for confirmation on all tools on an MCP server
policy.ask_user(mcp_server_cfg, handler=my_approval_handler)

# Ask for confirmation on a specific subset of tools on an MCP server
policy.ask_user(mcp_server_cfg, ["dangerous_tool"], handler=my_approval_handler)
```

### Wildcards

-   `policy.allow_all()`: Approves all tool calls. Equivalent to `allow("*")`.
-   `policy.deny_all()`: Denies all tool calls. Equivalent to `deny("*")`.

### Convenience Presets

-   `policy.confirm_run_command()`: Denies `run_command`, allows everything else.
    This is the **default** policy. Optionally accepts a `handler` to use
    `ask_user` instead of `deny`.
-   `policy.workspace_only(workspaces)`: Restricts `view_file`, `create_file`,
    and `edit_file` to paths within the given workspace directories.
    Automatically applied when `LocalAgentConfig.workspaces` is set.

## Predicates (Argument Checking)

You can use the `when` parameter to restrict policies based on tool arguments.
The predicate receives the tool arguments as a dictionary.

```python
from google.antigravity.hooks import policy

# Deny run_command if it contains 'rm'
policy.deny(
    "run_command",
    when=lambda args: "rm" in args.get("CommandLine", ""),
    name="deny_rm",
)
```

> [!CAUTION] If a predicate raises an exception during evaluation, the policy
> **fails closed** and treats it as a match (i.e., the decision for that policy
> applies).

## Minimal Safe Templates

### Deny by Default (Recommended for Production)

Start by denying everything and selectively allow safe tools.

```python
from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
from google.antigravity.hooks import policy

policies = [
    policy.deny_all(),
    policy.allow("view_file"),
    policy.allow("code_search"),
    policy.ask_user("run_command", handler=my_approval_handler),
]

config = LocalAgentConfig(
    system_instructions="You are a helpful assistant.",
    capabilities=CapabilitiesConfig(),  # Enables write tools
    policies=policies,
)
```

### Safe Default (No Configuration Needed)

The default `confirm_run_command()` policy is suitable for most use cases. Simply
create a config without specifying policies:

```python
from google.antigravity import Agent, LocalAgentConfig

# run_command is denied, all other tools allowed
config = LocalAgentConfig(
    system_instructions="You are a helpful assistant.",
)
```

### Allow All (Development Only)

Use only for local development where safety is not a concern.

```python
from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
from google.antigravity.hooks import policy

config = LocalAgentConfig(
    system_instructions="You are a helpful assistant.",
    capabilities=CapabilitiesConfig(),
    policies=[policy.allow_all()],
)
```

## Defense in Depth: OS-Level Command Sandboxing

Policies decide **whether** `run_command` runs; the OS sandbox confines **what a
command can do once it runs**. The two are complementary — a permissive policy
(e.g. `allow_all()`) can be paired with the sandbox so shell commands still
cannot touch anything outside the agent's workspace.

Enable it with `enable_sandbox` on `RunCommandConfig`:

```python
from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.hooks import policy

config = LocalAgentConfig(
    capabilities=types.CapabilitiesConfig(
        run_command_config=types.RunCommandConfig(enable_sandbox=True),
    ),
    # run_command must be allowed for the sandbox to apply.
    policies=[policy.allow_all()],
)
```

> [!IMPORTANT] `enable_sandbox` has no effect on platforms/environments where
> the sandbox is unavailable, and the SDK does not warn or raise in that case.
> Do not rely on it as your only safety boundary — combine it with policies
> (`deny` / `ask_user`) and verify the sandbox is available in your target
> environment.
