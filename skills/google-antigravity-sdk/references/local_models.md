# Running Agents with Local Models

This guide covers running Google Antigravity agents entirely on-device using
local models. Local execution does not require an API key.

There are two configuration classes for local model execution:

| Config Class | Backend | Auth Required | Use Case |
|---|---|---|---|
| `LiteRTAgentConfig` | Local LiteRT-LM | None | On-device models via LiteRT |
| `LocalOpenAIAgentConfig` | Local OpenAI-compat server | None | Ollama, LM Studio, etc. |

## LiteRTAgentConfig

`LiteRTAgentConfig` runs local models using the LiteRT-LM runtime. When
the agent starts, it spins up a local OpenAI-compatible loopback HTTP server
backed by the model checkpoint. All inference happens on-device.

> [!IMPORTANT]
> `LiteRTAgentConfig` requires `litert-lm>=0.15.0`. If `0.15.0` is not yet
> available on PyPI, install the nightly build: `pip install litert-lm-nightly`.

### Import

```python
from google.antigravity import Agent, LiteRTAgentConfig, LiteRTBackend
```

### Key Parameters

- `model_path` (str, **required**): Absolute path to a `.litertlm` model file.
  Tilde (`~`) is **not** expanded automatically — use `os.path.expanduser()` in
  Python or pass the full absolute path.
- `backend` (`'gpu'` | `'cpu'` | `'npu'`, default `'gpu'`): Hardware backend
  for inference. Use `LiteRTBackend.GPU`, `LiteRTBackend.CPU`, or
  `LiteRTBackend.NPU`.
- `compaction_config` (`CompactionConfig` | None, default None): Configure
  compaction and context limits, including `max_context_tokens` for KV-cache
  pre-allocation size. Set to `65536` to accommodate most local hardware setups.
  The model supports larger windows, but 64k balances capability with memory
  constraints. When unset, defaults to `4096` from model metadata.
- `capabilities` (`CapabilitiesConfig` | None, default None): Configure agent
  capabilities (subagents, tool allowlists, behavior mode).
- `enable_speculative_decoding` (bool, default False): Enable multi-token
  prediction for faster generation.
- `cache_dir` (str | None): Directory for compilation caching. Speeds up
  subsequent launches.
- `audio_backend` / `vision_backend`: Override the hardware backend for
  multimodal (audio / vision) processing.
- `port` (int, default 0): Port for the local loopback server. `0` selects a
  random available port.
- `download_if_missing` (bool, default False): Automatically download model
  weights if the `model_path` does not exist.

All standard `AgentConfig` parameters are also supported: `system_instructions`,
`capabilities`, `tools`, `policies`, `hooks`, `triggers`, `mcp_servers`,
`subagents`, `workspaces`, etc.

### Basic Example

For local execution, prefer `.lightweight()` to minimize prompt overhead,
restrict tools to core coding capabilities, disable subagents, and configure
context compaction for local models:

```python
import os

from google.antigravity import Agent, LiteRTAgentConfig

config = LiteRTAgentConfig(
    model_path=os.path.expanduser(
        "~/.litert-lm/models/gemma4-26b/model.litertlm"
    ),
).lightweight()
async with Agent(config=config) as agent:
    response = await agent.chat("Explain Python generators.")
    print(response)
```

### Full 64k Context Window

By default, the KV-cache is allocated for only 4096 tokens. To use the full
context window:

```python
config = LiteRTAgentConfig(
    model_path=os.path.expanduser(
        "~/.litert-lm/models/gemma4-26b/model.litertlm"
    ),
    compaction_config=CompactionConfig(max_context_tokens=65536),
)
```

> [!IMPORTANT] **Setting `max_context_tokens=65536` increases memory usage.**
> The model supports larger context windows, but 65536 is recommended to
> accommodate most local hardware setups. Ensure your device has sufficient RAM
> or VRAM.

### Speculative Decoding

Enable multi-token prediction for faster inference:

```python
config = LiteRTAgentConfig(
    model_path=os.path.expanduser(
        "~/.litert-lm/models/gemma4-26b/model.litertlm"
    ),
    enable_speculative_decoding=True,
)
```

### CPU Fallback

If no GPU is available, you can run on CPU. Set the backend explicitly or use
the environment variable:

```python
config = LiteRTAgentConfig(
    model_path=os.path.expanduser(
        "~/.litert-lm/models/gemma4-26b/model.litertlm"
    ),
    backend="cpu",
)
```

Or set the environment variable `ANTIGRAVITY_ALLOW_CPU=1` before running.

> [!WARNING] **CPU inference on large models is extremely slow.** This is only
> practical for small models or quick testing. For production workloads, use a
> GPU or consider a cloud-hosted model with `LocalAgentConfig`.

---

## LocalOpenAIAgentConfig

`LocalOpenAIAgentConfig` connects to any locally running OpenAI-compatible API
server, such as [Ollama](https://ollama.com) or
[LM Studio](https://lmstudio.ai).

> [!WARNING]
> Do **not** use `LocalOpenAIAgentConfig` to connect to `litert-lm serve`.
> Use `LiteRTAgentConfig` instead — it manages the LiteRT server lifecycle
> automatically. `LocalOpenAIAgentConfig` is only for external servers like
> Ollama or LM Studio that you start and manage independently.

### Import

```python
from google.antigravity import Agent, LocalOpenAIAgentConfig
```

### Key Parameters

- `model` (str | ModelTarget | None): The model identifier as recognized by the
  local server (e.g., `"llama3"`, `"gemma:7b"`).
- `base_url` (str | None): URL of the local server's OpenAI-compatible endpoint
  (e.g., `"http://localhost:11434/v1"` for Ollama).

All standard `AgentConfig` parameters are also supported: `system_instructions`,
`capabilities`, `tools`, `policies`, `hooks`, `triggers`, `mcp_servers`,
`subagents`, `workspaces`, etc.

### Example with Ollama

Start Ollama and pull a model first:

```bash
ollama pull gemma3:4b
```

Then create an agent (using `.lightweight()` for optimized prompt overhead and
core coding tools):

```python
from google.antigravity import Agent, LocalOpenAIAgentConfig

config = LocalOpenAIAgentConfig(
    model="gemma3:4b",
    base_url="http://localhost:11434/v1",
).lightweight()
async with Agent(config=config) as agent:
    response = await agent.chat("What is the capital of France?")
    print(response)
```

### Example with LM Studio

```python
from google.antigravity import Agent, LocalOpenAIAgentConfig

config = LocalOpenAIAgentConfig(
    model="gemma-4-26B-A4B-it",
    base_url="http://localhost:1234/v1",
).lightweight()
async with Agent(config=config) as agent:
    response = await agent.chat("Summarize this document.")
    print(response)
```

---

## Hardware & Platform Requirements

GPU detection is automatic. The SDK checks for available hardware acceleration
at startup and selects the appropriate backend. The detection logic is:

- **macOS**: Checks for Apple Silicon (`hw.optional.arm64`) → uses **Metal** via the `'gpu'` backend.
- **Linux**: Checks for `nvidia-smi` and CUDA libraries (`libcuda.so`) → uses **CUDA** via the `'gpu'` backend.
- **Windows**: Checks for `nvidia-smi` and CUDA libraries (`nvcuda.dll`). The LiteRT runtime also requires DirectX Shader Compiler components for GPU inference via WebGPU/Dawn.
- **NPU**: Available via `backend='npu'` where compatible hardware and drivers are present.
- **CPU fallback**: If no GPU is detected, set `backend='cpu'` or the environment variable `ANTIGRAVITY_ALLOW_CPU=1`.

### Decision Table

| Your Environment | Recommended Config | Backend |
|---|---|---|
| macOS + Apple Silicon | `LiteRTAgentConfig` | `'gpu'` (Metal) |
| Linux + NVIDIA GPU | `LiteRTAgentConfig` | `'gpu'` (CUDA) |
| Windows + NVIDIA GPU | `LiteRTAgentConfig` | `'gpu'` (CUDA) |
| Ollama / LM Studio running | `LocalOpenAIAgentConfig` | N/A |
| NPU-equipped device | `LiteRTAgentConfig` | `'npu'` |
| No GPU (CPU only) | `LiteRTAgentConfig` | `'cpu'` |

---

## Authentication

Local models do not require an API key or any external authentication.
