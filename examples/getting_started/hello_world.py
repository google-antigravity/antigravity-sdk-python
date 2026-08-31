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

"""Simple hello world example for Google Antigravity SDK.

This example demonstrates the simplest way to interact with an agent:
- Creating a configuration (and how to explicitly select a model).
- Using the Agent context manager.
- Sending a simple prompt and awaiting the full text response.

To run:
  python hello_world.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent produces a non-empty text response.
  3. The response contains "Hello World" or a close greeting variant.
"""

import asyncio

from google.antigravity import Agent
from google.antigravity import LocalAgentConfig


async def main() -> None:
  # To explicitly set the model, pass it to LocalAgentConfig:
  # config = LocalAgentConfig(model="gemini-3.8-flash")
  config = LocalAgentConfig()

  async with Agent(config) as my_agent:
    prompt = "Say 'Hello World!'"
    print(f"  User: {prompt}")

    response = await my_agent.chat(prompt)

    # Await the full aggregated text response.
    response_text = await response.text()
    print(f"  Agent: {response_text}")


if __name__ == "__main__":
  asyncio.run(main())
