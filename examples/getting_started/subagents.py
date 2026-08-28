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

"""Example demonstrating subagents and nested hierarchies in Google Antigravity SDK.

This script demonstrates three core subagent workflows:
  - Dynamic Self-Delegation: The agent dynamically spawns a clone of itself
    ("self") to delegate a heavy research task while keeping its own context
    window clean.
  - Custom Static Subagents: An agent configured with a dedicated static
    subagent definition ('code_reviewer') with scoped tools and custom system
    instructions.
  - Hierarchical Nested Subagents: A multi-tier delegation chain using
    max_subagent_depth and allowed_subagents scoping ('root' ->
    'lead_researcher'
    -> 'fact_checker').

To run:
  python subagents.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The dynamic subagent delegates researching the directory and produces a
     lesson plan.
  3. The 'code_reviewer' subagent audits mcp_server.py, producing warnings
     prefixed with '[AUDIT_WARNING]'.
  4. The subagent uses the 'get_reviewer_badge' tool to sign the report with
     'Senior-L3-Auditor-Badge'.
  5. The 'code_reviewer' subagent only has access to its allowlisted tool
     ('get_reviewer_badge') and cannot call unlisted root tools
     ('get_root_admin_secret').
  6. Subagent hook logs fire for all workflows, showing start/done events.
  7. The hierarchical delegation workflow successfully delegates from the root
     agent to 'lead_researcher', which further delegates to 'fact_checker',
     respecting max_subagent_depth=3 and allowed_subagents scoping.
"""

import asyncio
import logging
import pathlib
import sys
from typing import Any

from google.antigravity import Agent
from google.antigravity import LocalAgentConfig
from google.antigravity import types
from google.antigravity.hooks import hooks

_RESOURCES_DIR = pathlib.Path(__file__).resolve().parent.parent / "resources"
_subagent_active = False


@hooks.pre_tool_call_decide
async def log_pre_tool(data: types.ToolCall) -> types.HookResult:
  """Logs all tool calls for visibility."""
  global _subagent_active

  if data.name == types.BuiltinTools.START_SUBAGENT.value:
    _subagent_active = True
    print("\n  --- 🤖 [Hook] Spawning Subagent ---")
    print(f"  Arguments: {data.args}\n")
  else:
    indent = "    " if _subagent_active else "  "
    print(f"{indent}- [Start]: {data.name} (ID: {data.id})", flush=True)
  return types.HookResult(allow=True)


@hooks.post_tool_call
async def log_post_tool(data: Any) -> None:
  """Logs tool results."""
  global _subagent_active

  if data.name == types.BuiltinTools.START_SUBAGENT.value:
    _subagent_active = False
    print("\n  --- 🤖 [Hook] Subagent Finished ---")
    print(f"  Result: {data.result}\n")
  else:
    indent = "    " if _subagent_active else "  "
    print(f"{indent}- [Done]: {data.name} (ID: {data.id}) ✅", flush=True)


def get_reviewer_badge() -> str:
  """Returns the reviewer's official certification badge name."""
  return "Senior-L3-Auditor-Badge"


def get_root_admin_secret() -> str:
  """Returns the root admin super secret password for root administration only."""
  return "SUPER_SECRET_ROOT_PASSWORD_12345"


async def run_dynamic_subagent() -> None:
  """Runs a dynamic self-delegation research workflow."""
  print("\n=== Dynamic Subagent (Self Clone) ===")
  # Subagents are enabled by default on LocalAgentConfig.
  config = LocalAgentConfig(
      workspaces=[str(_RESOURCES_DIR)],
      hooks=[log_pre_tool, log_post_tool],
  )

  async with Agent(config) as my_agent:
    prompt = (
        "Use a subagent to research the example files in the resources"
        " directory. Delegate the task of listing and reading the files to the"
        " subagent, and then generate a lesson plan for me to learn more"
        " based on its findings."
    )
    print(f"  User: {prompt}")
    response = await my_agent.chat(prompt)
    response_text = await response.text()
    print(f"\n  Agent:\n{response_text}")


async def run_custom_static_subagent() -> None:
  """Runs a custom static subagent code-review audit workflow."""
  print("\n=== Custom Static Subagent ===")
  reviewer_subagent = types.SubagentConfig(
      name="code_reviewer",
      description="Audits source code files and reports missing docstrings.",
      system_instructions=(
          "You are a code reviewer. Read python files in the workspace and "
          "check if all function declarations have docstrings. For each "
          "function that is missing a docstring, output a warning prefixed "
          "with '[AUDIT_WARNING]'. "
          "CRITICAL: Every warning you output MUST start with "
          "'[AUDIT_WARNING]'. Use the 'get_reviewer_badge' tool to sign "
          "your final audit report with your official badge name. "
          "Also verify that you do not have access to any secret tools "
          "such as 'get_root_admin_secret' or any other root admin tools. "
          "State explicitly in your report that you only have access to "
          "your allowlisted reviewer tools and cannot call unlisted root "
          "tools. Output your report directly in your final response. Do not "
          "use the send_message tool to deliver it."
      ),
      tools=[get_reviewer_badge],
  )

  config = LocalAgentConfig(
      subagents=[reviewer_subagent],
      workspaces=[str(_RESOURCES_DIR)],
      tools=[get_reviewer_badge, get_root_admin_secret],
      hooks=[log_pre_tool, log_post_tool],
  )

  async with Agent(config) as my_agent:
    prompt = (
        "Ask the 'code_reviewer' subagent to review mcp_server.py, sign the"
        " report with their reviewer badge name, and verify whether they have"
        " access to the 'get_root_admin_secret' tool. Show me the exact"
        " warnings it produced verbatim (`[AUDIT_WARNING]`), the badge"
        " signature, and its verification that it cannot call"
        " 'get_root_admin_secret' or access root secrets."
    )
    print(f"  User: {prompt}")

    response = await my_agent.chat(prompt)
    response_text = await response.text()
    print(f"\n  Agent:\n{response_text}")

    # Print verification checks for developer reference:
    print("\n  === Verification Results ===")
    has_warning = "[AUDIT_WARNING]" in response_text
    print(
        f"  {'[PASS]' if has_warning else '[FAIL]'} Custom system prompt"
        " '[AUDIT_WARNING]' prefix check"
    )
    has_badge = "Senior-L3-Auditor-Badge" in response_text
    print(
        f"  {'[PASS]' if has_badge else '[FAIL]'} Allowlisted tool access"
        " ('Senior-L3-Auditor-Badge' signature) check"
    )
    no_secret = "SUPER_SECRET_ROOT_PASSWORD_12345" not in response_text
    print(
        f"  {'[PASS]' if no_secret else '[FAIL]'} Root secret isolation check"
        " (get_root_admin_secret not called)"
    )


async def run_nested_subagent_hierarchy() -> None:
  """Runs a 3-tier nested subagent hierarchy workflow."""
  print("\n=== Hierarchical Nested Subagents ===")
  # Tier 3 (leaf): A fact-checker that can read files but cannot spawn
  # further subagents.
  fact_checker = types.SubagentConfig(
      name="fact_checker",
      description=(
          "Reads specific files and verifies factual claims. Reports"
          " findings back to the caller."
      ),
      capabilities=types.SubagentCapabilities(
          enabled_tools=[
              types.BuiltinTools.VIEW_FILE,
              types.BuiltinTools.FIND_FILE,
          ],
      ),
  )

  # Tier 2 (middle): A lead researcher that can delegate to fact_checker.
  lead_researcher = types.SubagentConfig(
      name="lead_researcher",
      description=(
          "Researches a topic by reading files and delegating fact-checking"
          " to the 'fact_checker' subagent."
      ),
      capabilities=types.SubagentCapabilities(
          enabled_tools=[
              types.BuiltinTools.VIEW_FILE,
              types.BuiltinTools.FIND_FILE,
              types.BuiltinTools.LIST_DIR,
              types.BuiltinTools.START_SUBAGENT,
          ],
          allowed_subagents=["fact_checker"],
      ),
  )

  # Tier 1 (root): The main agent with a session-wide depth ceiling.
  config = LocalAgentConfig(
      subagents=[lead_researcher, fact_checker],
      workspaces=[str(_RESOURCES_DIR)],
      capabilities=types.CapabilitiesConfig(
          max_subagent_depth=3,
          allowed_subagents=["lead_researcher"],
      ),
      hooks=[log_pre_tool, log_post_tool],
  )

  async with Agent(config) as my_agent:
    prompt = (
        "Use the 'lead_researcher' subagent to investigate the MCP server"
        " and sample document in the workspace. The lead_researcher should"
        " delegate fact-checking of specific claims (such as the secret"
        " number in sample_doc.txt or math operations in mcp_server.py) to"
        " 'fact_checker'. Give me a summary of the server architecture and"
        " verified facts."
    )
    print(f"  User: {prompt}")
    response = await my_agent.chat(prompt)
    response_text = await response.text()
    print(f"\n  Agent:\n{response_text}")


async def main() -> None:
  # Configure logging
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  for h in root.handlers[:]:
    root.removeHandler(h)
  ch = logging.StreamHandler(sys.stderr)
  ch.setLevel(logging.INFO)
  ch.setFormatter(
      logging.Formatter("%(levelname)s:%(name)s:%(message)s")
  )
  root.addHandler(ch)

  await run_dynamic_subagent()
  await run_custom_static_subagent()
  await run_nested_subagent_hierarchy()


if __name__ == "__main__":
  asyncio.run(main())
