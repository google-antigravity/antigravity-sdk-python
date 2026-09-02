
# Conversation Compaction & Context Limits

This guide demonstrates how to configure background trajectory checkpointing and maximum context token ceilings using `CompactionConfig`.

---

## Overview

As an agent performs multi-turn tasks, its conversation trajectory grows. To prevent exceeding model token limits while retaining critical task history, Antigravity uses a two-stage sliding-window pipeline:

1. **Background Checkpointing**: A background model pre-computes cumulative trajectory summaries (checkpoints) at regular token intervals (`checkpoint_interval_tokens`). Checkpoint generation runs asynchronously and silently in parallel without modifying or truncating the active prompt. Every checkpoint is cumulative, summarizing history up to that point.
2. **Prompt Eviction (Compaction)**: When cumulative prompt tokens reach the context ceiling (`max_context_tokens`), the prompt snaps back to the latest completed checkpoint. Earlier checkpoints and older turns preceding the latest checkpoint are evicted from the prompt, while recent turns between the latest checkpoint and the current turn are preserved verbatim with full fidelity.

---

## Code Example

```python
import asyncio
from google.antigravity import Agent, LocalAgentConfig, types

# Configure compaction parameters on LocalAgentConfig
config = LocalAgentConfig(
    compaction_config=types.CompactionConfig(
        # Interval at which background checkpoints (summaries) are prepared.
        checkpoint_interval_tokens=40_000,
        # Maximum context window ceiling before older turns are evicted and
        # replaced by the latest background checkpoint.
        max_context_tokens=100_000,
    ),
)

async def main():
    async with Agent(config=config) as agent:
        response = await agent.chat(
            "Perform a deep multi-step analysis of our code repository."
        )
        print(await response.text())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Key Concepts

* **`CompactionConfig`**: Attached to `LocalAgentConfig(compaction_config=...)`, `LiteRTAgentConfig`, `LocalOpenAIAgentConfig`, or `AntigravityProdActorAgentConfig`.
* **`checkpoint_interval_tokens`**: Governs how often background trajectory checkpoints are generated. When omitted or `None`, the backend's default cadence is used.
* **`max_context_tokens`**: The hard ceiling for active conversation context sent to the model. When omitted or `None`, the backend's default context limit is used.
* **Validation Rule**: `checkpoint_interval_tokens` cannot exceed `max_context_tokens`. If both are configured, `checkpoint_interval_tokens <= max_context_tokens` is enforced.
* **Synchronous Fallback**: If `checkpoint_interval_tokens == max_context_tokens`, the harness performs immediate synchronous compaction when the threshold is crossed.
