# Built-in Tools Reference

In the `LocalAgentConfig` (used for local development), all built-in tools are
**enabled** by default. However, `run_command` is **denied** by the default
`confirm_run_command()` policy — all other tools are allowed. See
[Safety Policies](safety_policies.md) to customize this behavior.

> [!NOTE]
> Agents default to `agent_behavior=AgentBehavior.AUTONOMOUS`. When using
> interactive tools like `BuiltinTools.ASK_QUESTION`, configure
> `agent_behavior=AgentBehavior.INTERACTIVE` in `CapabilitiesConfig` so the
> agent actively collaborates with and prompts the user.

To override a built-in tool with a custom implementation, you can define a
custom tool with the same name. See
[Custom Tool Example](../examples/getting_started/custom_tool.md#overriding-built-in-tools)
for details.

The following table lists all built-in tools available in the SDK and their
descriptions.

| Tool Enum                     | Tool Name          | Description            |
| ----------------------------- | ------------------ | ---------------------- |
| `BuiltinTools.LIST_DIR`       | `list_directory`   | List directory         |
:                               :                    : contents.              :
| `BuiltinTools.SEARCH_DIR`     | `search_directory` | Search within          |
:                               :                    : directories.           :
| `BuiltinTools.FIND_FILE`      | `find_file`        | Find files by name.    |
| `BuiltinTools.VIEW_FILE`      | `view_file`        | View file contents.    |
| `BuiltinTools.FINISH`         | `finish`           | Finish and return      |
:                               :                    : output.                :
| `BuiltinTools.CREATE_FILE`    | `create_file`      | Create a new file.     |
| `BuiltinTools.EDIT_FILE`      | `edit_file`        | Edit an existing file. |
| `BuiltinTools.RUN_COMMAND`    | `run_command`      | Execute a shell        |
:                               :                    : command.               :
| `BuiltinTools.ASK_QUESTION`   | `ask_question`     | Ask user a question.   |
| `BuiltinTools.START_SUBAGENT` | `start_subagent`   | Invoke a subagent.     |
| `BuiltinTools.GENERATE_IMAGE` | `generate_image`   | Generate or edit       |
:                               :                    : images.                :
| `BuiltinTools.SEARCH_WEB`     | `search_web`       | Search the web for     |
:                               :                    : information.           :
| `BuiltinTools.READ_URL_CONTENT`| `read_url_content` | Fetch and read URL     |
:                               :                    : content.               :

> [!NOTE] Some production backends may require additional environment or
> filesystem configuration to support these tools.

> [!NOTE] `run_command` can be hardened by executing shell commands inside an
> OS-level sandbox. Set `enable_sandbox=True` on `RunCommandConfig` (nested
> under `CapabilitiesConfig.run_command_config`). See
> [Safety Policies](safety_policies.md#defense-in-depth-os-level-command-sandboxing)
> for details and caveats.
