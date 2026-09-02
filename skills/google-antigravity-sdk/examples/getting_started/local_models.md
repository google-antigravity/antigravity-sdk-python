# Getting Started: Running Agents with Local Models

## Overview

The Google Antigravity SDK supports running agents entirely on-device using local
models. No API key or cloud connectivity is required. This guide walks through
setup for two paths:

- **LiteRT** — optimized runtime for local models on-device.
- **OpenAI-compatible servers** — for Ollama, LM Studio, and similar tools.

---

## Path 1: LiteRT with Gemma

### Setup Steps

#### 1. Create a virtual environment (recommended)

Using a virtual environment avoids PATH issues and dependency conflicts:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install the SDK

```bash
pip install google-antigravity
```

#### 3. Install LiteRT dependencies

```bash
pip install litert-lm
```

> [!IMPORTANT]
> `LiteRTAgentConfig` requires `litert-lm>=0.15.0`. If `0.15.0` is not yet
> available on PyPI, install the nightly build instead:
> 
> ```bash
> pip install litert-lm-nightly
> ```

#### 4. Download a model checkpoint

Use the `litert-lm` CLI to import a model from Hugging Face:

```bash
litert-lm import \
  --from-huggingface-repo=litert-community/gemma-4-26B-A4B-it-litert-lm \
  gemma-4-26B-A4B-it-web.litertlm \
  gemma4-26b
```

> [!NOTE]
> This downloads approximately **16.8 GB** and registers the checkpoint at
> `~/.litert-lm/models/gemma4-26b/model.litertlm`.

> [!TIP]
> On macOS, if `litert-lm import` fails with an SSL certificate error, run:
> 
> ```bash
> pip install certifi
> export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
> ```
> Then retry the import command.

### Minimal Example

```python
import asyncio
import os

from google.antigravity import Agent, CompactionConfig, LiteRTAgentConfig


def compute_secret_hash(input_str: str) -> str:
    """Computes a secret hash encoding for the provided input string.

    Args:
        input_str: The text string to encode.
    """
    return input_str[::-1]


async def main():
    config = LiteRTAgentConfig(
        model_path=os.path.expanduser(
            "~/.litert-lm/models/gemma4-26b/model.litertlm"
        ),
        compaction_config=CompactionConfig(max_context_tokens=65536),
        tools=[compute_secret_hash],
    ).lightweight()
    async with Agent(config) as agent:
        response = await agent.chat("Compute the secret hash for 'Hello'.")
        async for token in response:
            print(token, end="", flush=True)
        print()


asyncio.run(main())
```

> [!IMPORTANT]
> `model_path` must be an absolute path.

### Key Configuration Notes

| Setting                 | Detail                                             |
| ----------------------- | -------------------------------------------------- |
| `.lightweight()` preset | Recommended for local execution. Automatically     |
:                         : selects core development tools, prunes system      :
:                         : prompt overhead, disables subagents, and           :
:                         : configures context compaction for local models.    :
| `compaction_config` (`max_context_tokens`) | Set to `CompactionConfig(max_context_tokens=65536)` to accommodate most local hardware setups. The model supports larger windows, but 64k balances capability with memory constraints. Default is `4096`. |
| GPU acceleration        | Auto-detected. Apple Silicon uses **Metal**;       |
:                         : Linux/Windows uses **CUDA**.                       :
| CPU-only fallback       | If no GPU is available, set `backend='cpu'` in the |
:                         : config or set the environment variable             :
:                         : `ANTIGRAVITY_ALLOW_CPU=1`.                         :

> [!NOTE]
> On Apple Silicon, the first execution compiles GPU graph shaders for Metal.
> This may take 1–2 minutes before token streaming starts. Subsequent launches
> use cached binaries and initialize in seconds.

---

## Path 2: OpenAI-Compatible Server (Ollama, LM Studio)

If you already run a local model server that exposes an OpenAI-compatible API,
point the SDK at it with `LocalOpenAIAgentConfig(...).lightweight()`:

```python
import asyncio
from google.antigravity import Agent, LocalOpenAIAgentConfig


async def main():
    config = LocalOpenAIAgentConfig(
        model="gemma2:27b",
        base_url="http://localhost:11434/v1",  # Ollama default
    ).lightweight()
    async with Agent(config) as agent:
        response = await agent.chat("Hello!")
        async for token in response:
            print(token, end="", flush=True)
        print()


asyncio.run(main())
```

> [!TIP]
> For **Ollama**, start the server with `ollama serve` and pull a model
> (`ollama pull gemma2:27b`) before running your agent. For **LM Studio**,
> enable the local server in the app settings and note the port it binds to.

> [!WARNING]
> Do **not** use `LocalOpenAIAgentConfig` to connect to `litert-lm serve`.
> Use `LiteRTAgentConfig` instead — it manages the LiteRT server lifecycle
> automatically. `LocalOpenAIAgentConfig` is for external servers like Ollama
> or LM Studio that you start and manage independently.

For the full comparison table and detailed configuration reference, see
[references/local_models.md](../references/local_models.md).
