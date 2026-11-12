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

r"""Example demonstrating Gemini Prioritized Inference via the SDK.

When priority inference is enabled, requests are routed to high-criticality
compute queues to deliver predictable second-level latency and strict
non-sheddable reliability.

If priority queue traffic exceeds dynamic rate limits, Google GenAI
automatically and gracefully downgrades requests to Standard tier processing
instead of failing with HTTP 429 or HTTP 503. This fallback mechanism ensures
higher reliability without always being charged more. This example demonstrates
how to configure priority inference and inspect the returned usage metadata to
programmatically monitor server-side downgrades.

Pricing Notice:
Priority tier requests are billed at a higher rate than Standard tier requests.
When overflow traffic is gracefully downgraded to Standard tier due to dynamic
rate limiting, those downgraded requests are billed at standard rates.
Please check the linked documentation for specific pricing, fallback thresholds,
and feature details:
https://ai.google.dev/gemini-api/docs/priority-inference

To run:
  python prioritized_inference.py
"""

import asyncio

from google.antigravity import Agent
from google.antigravity import GeminiAPIEndpoint
from google.antigravity import LocalAgentConfig
from google.antigravity import ModelTarget
from google.antigravity import types


async def main() -> None:
  # Configure priority inference by constructing a model endpoint and attaching
  # it to a ModelTarget.
  options = types.GeminiModelOptions(service_tier=types.ServiceTier.PRIORITY)
  endpoint = GeminiAPIEndpoint(options=options)
  config = LocalAgentConfig(
      model=ModelTarget(
          name="gemini-4-hydrogen",
          endpoint=endpoint,
      ),
  )

  async with Agent(config=config) as agent:
    prompt = "Explain quantum computing in one sentence."
    print(f"Sending prompt with priority tier: {prompt}")
    response = await agent.chat(prompt)
    print(f"Agent: {await response.text()}")

    # Inspect usage metadata to check if server-side downgrade occurred.
    usage = response.usage_metadata
    if usage and usage.service_tier:
      print(f"Served Service Tier: {usage.service_tier}")
      if usage.service_tier == types.ServiceTier.STANDARD:
        print(
            "Notice: Request was gracefully downgraded from priority to"
            " standard tier due to dynamic rate limiting."
        )
      elif usage.service_tier == types.ServiceTier.PRIORITY:
        print("Success: Request was served on high-criticality priority tier.")
    else:
      print("Served Service Tier: Unspecified / Not returned")


if __name__ == "__main__":
  asyncio.run(main())
