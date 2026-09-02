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

"""Example demonstrating OS-level sandboxing of terminal commands.

By default the agent's ``run_command`` tool executes shell commands directly on
the host. When you opt in with ``RunCommandConfig(enable_sandbox=True)``,
``run_command`` is instead executed inside the OS-level sandbox, which
confines what the command can touch.

``run_command`` must be allowed for the sandbox to matter, so this example pairs
the flag with ``policy.allow_all()``.

.. warning::

   ``allow_all()`` grants unrestricted tool access, including shell execution.
   The sandbox is the safety boundary here; only run this in a trusted
   environment.

The demo asks the agent to write a file *outside* its workspace (into
``$HOME``). With the sandbox enabled the write is blocked; with it disabled the
same command would succeed.

To run:
  python sandboxing.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent produces a non-empty text response.
  3. With the sandbox enabled, the response reports that the out-of-workspace
     write was blocked, i.e. the file was NOT created outside the workspace.
"""

import asyncio
import os

from google.antigravity import Agent
from google.antigravity import CapabilitiesConfig
from google.antigravity import LocalAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.types import RunCommandConfig

# A path deliberately *outside* the agent's workspace. A normal (unsandboxed)
# process owned by the user can write here; the exebox sandbox denies it.
_ESCAPE_PROBE_PATH = os.path.join(
    os.path.expanduser("~"), "agy_sandbox_escape_probe.txt"
)


async def run_probe(*, enable_sandbox: bool) -> str:
  """Runs the escape-probe command under an agent with the given sandbox flag.

  Args:
    enable_sandbox: Whether to execute run_command inside the OS sandbox.

  Returns:
    The agent's final text response.
  """
  config = LocalAgentConfig(
      capabilities=CapabilitiesConfig(
          run_command_config=RunCommandConfig(enable_sandbox=enable_sandbox),
      ),
      # allow_all() is required so run_command executes without confirmation;
      # the sandbox (not a confirmation prompt) is what contains it.
      policies=[policy.allow_all()],
  )

  # A concrete command that writes to a throwaway path outside the sandbox
  # boundary, so the block is attributable to the sandbox rather than to the
  # model declining on its own.
  prompt = (
      "Run exactly this shell command and then tell me whether it succeeded, "
      "quoting any error output verbatim:\n"
      f"  echo 'sandbox escape attempt' > '{_ESCAPE_PROBE_PATH}'"
  )

  async with Agent(config) as agent:
    response = await agent.chat(prompt)
    return await response.text()


async def main() -> None:
  # Best-effort clean slate so a leftover file from a prior unsandboxed run
  # doesn't confuse the result.
  try:
    os.remove(_ESCAPE_PROBE_PATH)
  except FileNotFoundError:
    pass

  print("=== Sandbox ENABLED: out-of-workspace write should be blocked ===")
  print(f"  User: attempt to write to {_ESCAPE_PROBE_PATH}")
  sandboxed_response = await run_probe(enable_sandbox=True)
  print(f"  Agent: {sandboxed_response}")

  escaped = os.path.exists(_ESCAPE_PROBE_PATH)
  print()
  if escaped:
    # enable_sandbox has no effect where the exebox sandbox is unavailable, and
    # the SDK neither warns nor raises in that case -- so surface it loudly here
    # rather than letting the write pass silently.
    print(
        "  WARNING: file WAS created -> the write was NOT sandboxed. "
        "enable_sandbox has no effect where the exebox sandbox is unavailable. "
        "Verify the sandbox is available in your environment before relying "
        "on it."
    )
    os.remove(_ESCAPE_PROBE_PATH)
  else:
    print(
        "  RESULT: file was NOT created -> the OS sandbox denied the "
        "out-of-workspace write, as expected."
    )

  # To see the contrasting baseline (the same command succeeding without the
  # sandbox), flip the flag -- kept commented out so the example is safe by
  # default:
  #
  #   baseline = await run_probe(enable_sandbox=False)
  #   print(baseline)  # the file is created under $HOME
  #   os.remove(_ESCAPE_PROBE_PATH)


if __name__ == "__main__":
  asyncio.run(main())
