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

"""Example demonstrating Vertex AI authentication in Google Antigravity SDK.

This example demonstrates how to connect to Vertex AI using either:
1. Express Mode: API key authentication (vertex=True, api_key="...").
2. Standard Mode: Google Cloud project and location (vertex=True, project="...",
   location="...").

To run:
  # Option 1: Express Mode (API Key)
  VERTEX_API_KEY="your-api-key" python vertex.py

  # Option 2: Standard Mode (Project & Location with ADC)
  GOOGLE_CLOUD_PROJECT="your-project" python vertex.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent produces a non-empty text response.
"""

import argparse
import asyncio
import os

from google.antigravity import Agent
from google.antigravity import LocalAgentConfig


async def main() -> None:
  parser = argparse.ArgumentParser(
      description="Vertex AI authentication example."
  )
  parser.add_argument(
      "--api_key",
      default=None,
      help="API key for Vertex Express Mode (defaults to VERTEX_API_KEY).",
  )
  parser.add_argument(
      "--project",
      default=None,
      help="Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT).",
  )
  parser.add_argument(
      "--location",
      default=None,
      help=(
          "Google Cloud location (defaults to GOOGLE_CLOUD_LOCATION or"
          " us-central1)."
      ),
  )
  args = parser.parse_args()

  api_key = args.api_key or os.environ.get("VERTEX_API_KEY")
  project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
  location = (
      args.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
  )

  if api_key and project:
    raise ValueError(
        "Cannot specify both api_key (Express Mode) and project (Standard"
        " Mode). They are mutually exclusive."
    )

  if api_key:
    print("Authenticating via Vertex AI Express Mode (API Key)...")
    config = LocalAgentConfig(
        vertex=True,
        api_key=api_key,
    )
  elif project:
    print(
        f"Authenticating via Vertex AI Standard Mode (Project: {project},"
        f" Location: {location})..."
    )
    config = LocalAgentConfig(
        vertex=True,
        project=project,
        location=location,
    )
  else:
    raise ValueError(
        "No Vertex AI credentials provided.\n"
        "Provide an API key via --api_key or VERTEX_API_KEY (Express Mode),\n"
        "or provide project/location via --project/--location or"
        " GOOGLE_CLOUD_PROJECT (Standard Mode)."
    )

  async with Agent(config) as agent:
    prompt = "Tell me a software engineering joke."
    print(f"  User: {prompt}")
    response = await agent.chat(prompt)
    print(f"  Agent: {await response.text()}\n")


if __name__ == "__main__":
  asyncio.run(main())

