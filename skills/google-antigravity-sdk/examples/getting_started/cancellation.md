
# Programmatic and Task Cancellation

This guide demonstrates how to programmatically interrupt in-progress generation turns.

---

## Programmatic Turn-Level Cancellation

Exposed directly via `ChatResponse.cancel()`. This cleanly aborts the active execution turn on the backend (stopping inference and subagents) while leaving the conversational session fully open and preserving all steps completed prior to cancellation in history.

```python
import asyncio
from google.antigravity import Agent, LocalAgentConfig, types

async with Agent(LocalAgentConfig()) as agent:
    response = await agent.chat("Write a very long story about cancellation.")
    stream_task = asyncio.create_task(response.text())

    # In another concurrent coroutine or after a short timeout:
    if generation_needs_to_stop():
        await response.cancel()  # Halts active inference on the backend cleanly

    try:
        print(await stream_task)
    except types.AntigravityCancelledError:
        print("Turn cancelled programmatically.")
    except asyncio.CancelledError:
        raise
```

---

## Key Concepts

*   **`response.cancel()`**: Transient, scoped strictly to the active turn. Calling this method aborts active backend execution and triggers any active stream generator cursor to raise `types.AntigravityCancelledError`. Safe to call multiple times; once the stream completes, calling `.cancel()` is a safe, silent no-op.
*   **`types.AntigravityCancelledError`**: A custom SDK exception raised when the active turn is aborted. Importantly, it **inherits from `asyncio.CancelledError`** (and thus `BaseException`).
    - It cleanly bypasses generic `except Exception` blocks, preventing cancellation from being swallowed or mislogged as execution errors.
    - Place it first in your exception catch hierarchy to explicitly handle SDK programmatic cancels separately from native task cancellations.
*   **State Preservation**: History (completed steps, token usage) generated up to the cancellation boundary remains fully preserved inside `Conversation.history` for state persistence and resume flows.
