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

"""Conversation compaction and context limits example for Google Antigravity SDK.

This example demonstrates how to configure background trajectory checkpointing
and maximum context token ceilings using CompactionConfig.

To run:
  python compaction.py
"""

import asyncio

from google.antigravity import Agent
from google.antigravity import LocalAgentConfig
from google.antigravity import types


async def main() -> None:
  config = LocalAgentConfig(
      compaction_config=types.CompactionConfig(
          # Interval at which background checkpoints (summaries) are prepared.
          checkpoint_interval_tokens=40_000,
          # Maximum context window ceiling before older turns are evicted and
          # replaced by the latest background checkpoint.
          max_context_tokens=100_000,
      ),
  )

  async with Agent(config) as my_agent:
    prompt = "Explain in one sentence what context compaction accomplishes."
    print(f"  User: {prompt}")

    response = await my_agent.chat(prompt)
    response_text = await response.text()
    print(f"  Agent: {response_text}")


if __name__ == "__main__":
  asyncio.run(main())
