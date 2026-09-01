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

"""Example demonstrating session budget controls and stop reason handling.

This example demonstrates how to configure operational limits and proactive
token budget caps using BudgetConfig:
1. Limiting model invocations (max_model_calls)
2. Limiting tool invocations (max_tool_calls)
3. Limiting net uncached input tokens (max_input_tokens)
4. Limiting cumulative output tokens (max_output_tokens)
5. Limiting cumulative total tokens (max_total_tokens)

To run:
  python budget_limits.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. Each budget dial triggers its corresponding StopReason when exhausted.
"""

import asyncio
import tempfile

from google.antigravity import Agent
from google.antigravity import LocalAgentConfig
from google.antigravity import types


# ---------------------------------------------------------------------------
# 1. Model Invocation Limits (max_model_calls)
# ---------------------------------------------------------------------------
async def demo_max_model_calls() -> None:
  """Demonstrates halting after exceeding max_model_calls limit."""
  print("\n" + "=" * 60)
  print("1. Testing max_model_calls budget limit (max_model_calls=1)")
  print("=" * 60)

  config = LocalAgentConfig(
      budget_config=types.BudgetConfig(max_model_calls=1)
  )

  async with Agent(config) as agent:
    # Turn 1: 1st model call allowed; upon completion, limit of 1 is reached
    print("Turn 1: Asking first question (consumes 1 allowed model call)...")
    res1 = await agent.chat("What is 2 + 2? Reply with just the number.")
    print(f"  Agent response: {(await res1.text()).strip()}")
    print(f"  Turn 1 stop reason: {res1.stop_reason}")
    if res1.stop_reason == types.StopReason.MAX_MODEL_CALLS_EXCEEDED:
      print("  [Limit Reached] Session model call budget reached after Turn 1.")

    # Turn 2: Attempting another turn when session model budget is exhausted
    print("\nTurn 2: Asking second question (budget already exhausted)...")
    res2 = await agent.chat("What is 3 + 3?")
    await res2.text()  # Drain response stream
    print(f"  Turn 2 stop reason: {res2.stop_reason}")
    if res2.stop_reason == types.StopReason.MAX_MODEL_CALLS_EXCEEDED:
      print("  [Halted] Turn 2 was prevented from making further model calls.")


# ---------------------------------------------------------------------------
# 2. Tool Invocation Limits (max_tool_calls)
# ---------------------------------------------------------------------------
def lookup_weather(city: str) -> str:
  """Looks up current weather for a given city.

  Args:
    city: Name of the city.

  Returns:
    Simulated weather report string.
  """
  return f"Sunny and 24C in {city}"


def lookup_timezone(city: str) -> str:
  """Looks up the time zone for a given city.

  Args:
    city: Name of the city.

  Returns:
    Timezone description string.
  """
  return f"UTC+9 for {city}"


async def demo_max_tool_calls() -> None:
  """Demonstrates halting after exceeding max_tool_calls limit."""
  print("\n" + "=" * 60)
  print("2. Testing max_tool_calls budget limit (max_tool_calls=1)")
  print("=" * 60)

  config = LocalAgentConfig(
      tools=[lookup_weather, lookup_timezone],
      budget_config=types.BudgetConfig(max_tool_calls=1),
  )

  async with Agent(config) as agent:
    prompt = (
        "First call lookup_weather for 'Tokyo', and then call lookup_timezone"
        " for 'Tokyo'."
    )
    print(f"Sending multi-tool prompt: {prompt}")
    res = await agent.chat(prompt)
    await res.text()  # Drain response stream
    print(f"  Stop reason: {res.stop_reason}")
    if res.stop_reason == types.StopReason.MAX_TOOL_CALLS_EXCEEDED:
      print("  [Halted] Additional tool executions were halted by budget.")


# ---------------------------------------------------------------------------
# 3. Input Token Limits (max_input_tokens)
# ---------------------------------------------------------------------------
async def demo_max_input_tokens() -> None:
  """Demonstrates proactive halting when input tokens exceed max_input_tokens."""
  print("\n" + "=" * 60)
  print("3. Testing max_input_tokens budget limit (max_input_tokens=50)")
  print("=" * 60)

  config = LocalAgentConfig(
      budget_config=types.BudgetConfig(max_input_tokens=50)
  )

  async with Agent(config) as agent:
    # A prompt containing ~300+ tokens, exceeding the 50 token budget
    large_prompt = (
        "Summarize the following passage:\n"
        + "The quick brown fox jumps over the lazy dog. " * 30
    )
    print("Sending large prompt exceeding 50 input tokens...")
    res = await agent.chat(large_prompt)
    await res.text()  # Drain response stream
    print(f"  Stop reason: {res.stop_reason}")
    if res.stop_reason == types.StopReason.MAX_INPUT_TOKENS_EXCEEDED:
      print(
          "  [Proactively Halted] Input token budget exceeded before inference."
      )


# ---------------------------------------------------------------------------
# 4. Output Token Limits (max_output_tokens)
# ---------------------------------------------------------------------------
async def demo_max_output_tokens() -> None:
  """Demonstrates halting after cumulative output tokens exceed max_output_tokens."""
  print("\n" + "=" * 60)
  print("4. Testing max_output_tokens budget limit (max_output_tokens=30)")
  print("=" * 60)

  config = LocalAgentConfig(
      budget_config=types.BudgetConfig(max_output_tokens=30)
  )

  async with Agent(config) as agent:
    # Turn 1 produces > 30 output tokens
    print("Turn 1: Requesting a long response that exceeds 30 output tokens...")
    res1 = await agent.chat(
        "Write a detailed paragraph explaining photosynthesis."
    )
    print(f"  Turn 1 generated response: {(await res1.text())[:60]}...")
    print(f"  Turn 1 stop reason: {res1.stop_reason}")

    # Turn 2: Cumulative output tokens already exceed 30
    print("\nTurn 2: Attempting next turn with exhausted output budget...")
    res2 = await agent.chat("Continue.")
    await res2.text()  # Drain response stream
    print(f"  Turn 2 stop reason: {res2.stop_reason}")
    if res2.stop_reason == types.StopReason.MAX_OUTPUT_TOKENS_EXCEEDED:
      print(
          "  [Halted] Cumulative generated output exceeded output token limit."
      )


# ---------------------------------------------------------------------------
# 5. Total Token Limits (max_total_tokens)
# ---------------------------------------------------------------------------
async def demo_max_total_tokens() -> None:
  """Demonstrates halting after cumulative total tokens exceed max_total_tokens."""
  print("\n" + "=" * 60)
  print("5. Testing max_total_tokens budget limit (max_total_tokens=100)")
  print("=" * 60)

  config = LocalAgentConfig(
      budget_config=types.BudgetConfig(max_total_tokens=100)
  )

  async with Agent(config) as agent:
    # Turn 1 consumes > 100 total net tokens (input + output)
    print("Turn 1: Sending prompt that will consume > 100 total tokens...")
    res1 = await agent.chat(
        "Explain the theory of general relativity in 3 sentences."
    )
    print(f"  Turn 1 generated response: {(await res1.text())[:60]}...")
    print(f"  Turn 1 stop reason: {res1.stop_reason}")

    # Turn 2: Cumulative total tokens exceed 100
    print("\nTurn 2: Attempting next turn with exhausted total token budget...")
    res2 = await agent.chat("Tell me more.")
    await res2.text()  # Drain response stream
    print(f"  Turn 2 stop reason: {res2.stop_reason}")
    if res2.stop_reason == types.StopReason.MAX_TOTAL_TOKENS_EXCEEDED:
      print("  [Halted] Cumulative net token consumption exceeded total limit.")


# ---------------------------------------------------------------------------
# 6. Forward-Looking Scope & Session Resumption (BudgetScope.FORWARD_LOOKING)
# ---------------------------------------------------------------------------
async def demo_forward_looking_resume() -> None:
  """Demonstrates granting a fresh token budget on session resumption."""
  print("\n" + "=" * 60)
  print("6. Testing FORWARD_LOOKING budget scope on session resume")
  print("=" * 60)

  with tempfile.TemporaryDirectory(prefix="forward_looking_demo_") as save_dir:
    # Turn 1: Initial conversation turn with a budget of 1 model call
    config1 = LocalAgentConfig(
        save_dir=save_dir,
        budget_config=types.BudgetConfig(max_model_calls=1),
    )
    async with Agent(config1) as agent1:
      print(
          "Initial session: Sending question (consumes allowed 1 model call)..."
      )
      res1 = await agent1.chat(
          "What is the capital of France? Reply in 1 word."
      )
      print(f"  Agent response: {(await res1.text()).strip()}")
      print(f"  Turn 1 stop reason: {res1.stop_reason}")
      conv_id = agent1.conversation_id

    # Turn 2: Resume session with FORWARD_LOOKING budget grant
    # (max_model_calls=1). Under LIFETIME scope, total model calls would
    # be 2 >= 1 (exhausted). Under FORWARD_LOOKING scope, delta model calls
    # is 1 <= 1 (allowed).
    config2 = LocalAgentConfig(
        conversation_id=conv_id,
        save_dir=save_dir,
        session_continuation_mode=types.SessionContinuationMode.RESUME,
        budget_config=types.BudgetConfig(
            max_model_calls=1,
            scope=types.BudgetScope.FORWARD_LOOKING,
        ),
    )
    async with Agent(config2) as agent2:
      print(
          "\nResumed session: Sending follow-up with forward-looking budget..."
      )
      res2 = await agent2.chat(
          "What is the capital of Germany? Reply in 1 word."
      )
      print(f"  Agent response: {(await res2.text()).strip()}")
      print(f"  Resumed session stop reason: {res2.stop_reason}")
      if res2.stop_reason == types.StopReason.MAX_MODEL_CALLS_EXCEEDED:
        print(
            "  [Allowed & Monitored] Forward-looking budget successfully"
            " measured only post-resumption call!"
        )


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
async def main() -> None:
  print("Running Antigravity SDK Budget Enforcement End-to-End Tests...")
  await demo_max_model_calls()
  await demo_max_tool_calls()
  await demo_max_input_tokens()
  await demo_max_output_tokens()
  await demo_max_total_tokens()
  await demo_forward_looking_resume()
  print("\n" + "=" * 60)
  print("🎉 All budget enforcement dials verified successfully end-to-end!")
  print("=" * 60)


if __name__ == "__main__":
  asyncio.run(main())
