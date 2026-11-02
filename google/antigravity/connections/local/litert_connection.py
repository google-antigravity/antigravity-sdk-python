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

"""LiteRT Local connection strategy."""

import asyncio
import json
import logging
import os
import shutil
import sys
import threading
from typing import Any
import urllib.request

from google.antigravity import types
from google.antigravity.connections.local import litert_connection_config
from google.antigravity.connections.local import litert_server
from google.antigravity.connections.local import local_openai_connection

try:
  # pylint: disable=g-import-not-at-top
  import litert_lm  # type: ignore[import-error]

  _LITERT_AVAILABLE = True
except ImportError:
  litert_lm = None
  _LITERT_AVAILABLE = False

_WARMUP_REQUEST_TIMEOUT_SECONDS = 120

# Default token budget allocated for model thinking and reasoning.
_DEFAULT_THINKING_TOKEN_BUDGET = 8192


def _detect_default_vision_backend(
    model_path: str,
) -> litert_connection_config.LiteRTBackend | None:
  """Detects whether model supports vision and returns default vision backend."""
  # Requires the ModelInfo API in litert-lm 0.19.0 or above.
  if litert_lm is None or not hasattr(litert_lm, "ModelInfo"):
    return None

  try:
    with litert_lm.ModelInfo(model_path) as model_info:
      input_modalities = getattr(model_info, "input_modalities", None)
      if getattr(input_modalities, "vision", None) is not True:
        return None
      supported = None
      if hasattr(model_info, "supported_backends_for_modality") and hasattr(
          litert_lm, "LiteRtLmModality"
      ):
        try:
          supported = model_info.supported_backends_for_modality(
              litert_lm.LiteRtLmModality.VISION
          )
        except Exception:  # pylint: disable=broad-exception-caught
          supported = None
      if isinstance(supported, list):
        for backend in supported:
          normalized_backend = str(backend).strip().lower()
          if normalized_backend == "gpu":
            return litert_connection_config.LiteRTBackend.GPU
          if normalized_backend == "cpu":
            return litert_connection_config.LiteRTBackend.CPU
      return litert_connection_config.LiteRTBackend.CPU
  except Exception:  # pylint: disable=broad-exception-caught
    logging.warning(
        "Failed to inspect model info for vision detection at %s",
        model_path,
        exc_info=True,
    )
    return None


class LiteRTConnectionStrategy(
    local_openai_connection.LocalOpenAIConnectionStrategy
):
  """Strategy establishing connection to a local LiteRT loopback API server."""

  def __init__(
      self,
      *,
      model_path: str,
      backend: litert_connection_config.LiteRTBackend,
      enable_speculative_decoding: bool = False,
      cache_dir: str | None = None,
      audio_backend: litert_connection_config.LiteRTBackend | None = None,
      vision_backend: litert_connection_config.LiteRTBackend | None = None,
      port: int = 0,
      download_if_missing: bool = False,
      compaction_config: types.CompactionConfig | None = None,
      **kwargs: Any,
  ):
    self._model_path = model_path
    self._backend = backend
    self._enable_speculative_decoding = enable_speculative_decoding
    self._cache_dir = cache_dir
    self._audio_backend = audio_backend
    self._vision_backend = vision_backend
    self._litert_port = port
    self._download_if_missing = download_if_missing
    self._max_kv_cache_tokens = (
        litert_connection_config._DEFAULT_MAX_KV_CACHE_TOKENS
    )
    self._max_output_tokens = (
        litert_connection_config._DEFAULT_MAX_OUTPUT_TOKENS
    )
    self._thinking_token_budget = _DEFAULT_THINKING_TOKEN_BUDGET

    # Strategy Context Lifetimes
    self._engine = None
    self._engine_context = None
    self._openai_server = None
    self._openai_server_thread = None

    self._configure_litert_logging()

    if compaction_config is not None:
      effective_compaction = compaction_config
    else:
      effective_compaction = (
          litert_connection_config.derive_litert_compaction_config(
              max_kv_cache_tokens=self._max_kv_cache_tokens,
              max_output_tokens=self._max_output_tokens,
          )
      )

    super().__init__(
        base_url="",
        model_name=os.path.basename(model_path),
        compaction_config=effective_compaction,
        **kwargs,
    )

  def _configure_litert_logging(self) -> None:
    """Configures litert_lm C++ log severity level according to standard Python logging."""
    if not _LITERT_AVAILABLE or litert_lm is None:
      return
    if not hasattr(litert_lm, "set_min_log_severity"):
      return

    try:
      # If Python logging is set to DEBUG level or lower (e.g.
      # logging.basicConfig(level=logging.DEBUG)), enable verbose LiteRT C++
      # logs. Otherwise silence them by default.
      is_debug = logging.getLogger().isEnabledFor(
          logging.DEBUG
      ) or logging.getLogger("google.antigravity").isEnabledFor(logging.DEBUG)

      if is_debug:
        verbose_level = (
            getattr(litert_lm.LogSeverity, "VERBOSE", 0)
            if hasattr(litert_lm, "LogSeverity")
            else 0
        )
        litert_lm.set_min_log_severity(int(verbose_level))  # Safe cast for dynamic LogSeverity enum value  # pyrefly: ignore[bad-argument-type]
      else:
        silent_level = (
            getattr(litert_lm.LogSeverity, "SILENT", 1000)
            if hasattr(litert_lm, "LogSeverity")
            else 1000
        )
        litert_lm.set_min_log_severity(int(silent_level))  # Safe cast for dynamic LogSeverity enum value  # pyrefly: ignore[bad-argument-type]
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Failed to configure LiteRT min log severity: %s", e)

  @property
  def _openai_server_url(self) -> str:
    return self._base_url

  @_openai_server_url.setter
  def _openai_server_url(self, value: str) -> None:
    self._base_url = value

  def _validate_connection(self) -> None:
    """Overrides parent Gemini API key check."""
    if not os.path.exists(self._model_path):
      raise types.AntigravityValidationError(
          f"LiteRT model path does not exist: {self._model_path}"
      )

  def _validate_hardware(self) -> None:
    """Verifies hardware acceleration availability (warning checks to prevent false negatives)."""
    if os.environ.get("ANTIGRAVITY_ALLOW_CPU") == "1" or self._backend in (
        litert_connection_config.LiteRTBackend.CPU,
        litert_connection_config.LiteRTBackend.NPU,
    ):
      return

    has_gpu = _check_gpu_acceleration_available()

    if not has_gpu:
      logging.warning(
          "GPU acceleration hardware (Metal or CUDA) was not verified on this"
          " host. Quantized local Gemma execution on CPU can be extremely slow."
          " If initialization fails or is extremely slow, consider setting"
          " environment variable ANTIGRAVITY_ALLOW_CPU=1 or backend='cpu' in"
          " config."
      )

  async def __aenter__(self) -> None:
    self._validate_hardware()
    self._validate_connection()

    if not _LITERT_AVAILABLE or litert_lm is None:
      raise RuntimeError(
          "The 'litert-lm-api' PyPI package is required. "
          "Install it using: pip install litert-lm-api"
      )

    self._configure_litert_logging()

    def map_backend(
        b_enum: litert_connection_config.LiteRTBackend | None,
    ) -> Any:
      if not b_enum:
        return litert_lm.Backend.CPU()
      if b_enum == litert_connection_config.LiteRTBackend.GPU:
        return litert_lm.Backend.GPU()
      if b_enum == litert_connection_config.LiteRTBackend.NPU:
        return litert_lm.Backend.NPU()
      return litert_lm.Backend.CPU()

    engine_backend = map_backend(self._backend)
    engine_audio = (
        map_backend(self._audio_backend) if self._audio_backend else None
    )
    vision_backend = self._vision_backend or _detect_default_vision_backend(
        self._model_path
    )
    engine_vision = map_backend(vision_backend) if vision_backend else None

    try:
      # Load engine context manager
      self._engine = litert_lm.Engine(
          self._model_path,
          backend=engine_backend,
          enable_speculative_decoding=self._enable_speculative_decoding,
          cache_dir=self._cache_dir,
          audio_backend=engine_audio,
          vision_backend=engine_vision,
          max_num_tokens=self._max_kv_cache_tokens,
          # Since the Engine is used in a stateless OpenAI server, we can enable
          # "use_ringbuffers_local_attention" to support larger context length
          # with limited memory.
          use_ringbuffers_local_attention=True,
      )
      self._engine_context = self._engine.__enter__()

      # Bind HTTPServer directly to port 0 (Safe free port allocation)
      logging.debug(
          "LiteRTConnectionStrategy __aenter__: Instantiating"
          " LiteRTOpenAIServer"
      )
      self._openai_server = litert_server.LiteRTOpenAIServer(
          ("127.0.0.1", self._litert_port),
          litert_server.LiteRTOpenAIHandler,
          engine=self._engine_context,
          model_name=self._model_name,
          max_output_tokens=self._max_output_tokens,
          thinking_token_budget=self._thinking_token_budget,
      )
      addr = self._openai_server.server_address
      host = addr[0]
      actual_port = addr[1]
      self._openai_server_url = f"http://{host}:{actual_port}"
      self._base_url = self._openai_server_url
      logging.debug(
          "LiteRTConnectionStrategy __aenter__: server url is %s",
          self._openai_server_url,
      )

      logging.debug(
          "LiteRTConnectionStrategy __aenter__: Starting server thread"
      )
      self._openai_server_thread = threading.Thread(
          target=self._openai_server.serve_forever, daemon=True
      )
      self._openai_server_thread.start()
      logging.debug(
          "LiteRTConnectionStrategy __aenter__: Server thread started"
      )

      # Non-blocking event loop ping polling via executor (proxy-less urlopen)
      logging.debug(
          "LiteRTConnectionStrategy __aenter__: Starting health check loop"
      )
      loop = asyncio.get_running_loop()
      health_ok = False

      def _ping():
        logging.debug("LiteRTConnectionStrategy __aenter__: _ping start")
        with _urlopen_no_proxy(
            f"{self._openai_server_url}/v1/models", timeout=0.5
        ) as r:
          # Drain the response body so the socket closes cleanly without
          # triggering abortive TCP RST frames (WinError 10054) on Windows.
          r.read()
          status = r.status
          logging.debug(
              "LiteRTConnectionStrategy __aenter__: _ping response status: %s",
              status,
          )
          return status == 200

      for i in range(60):
        try:
          logging.debug(
              "LiteRTConnectionStrategy __aenter__: ping iteration %s", i
          )

          status_ok = await loop.run_in_executor(None, _ping)
          logging.debug(
              "LiteRTConnectionStrategy __aenter__: _ping outcome status_ok=%s",
              status_ok,
          )
          if status_ok:
            health_ok = True
            break
        # pylint: disable=broad-exception-caught
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.debug(
              "LiteRTConnectionStrategy __aenter__: ping exception: %s", e
          )
        await asyncio.sleep(0.5)

      if not health_ok:
        logging.debug(
            "LiteRTConnectionStrategy __aenter__: Health check FAILED"
        )
        raise RuntimeError("Loopback HTTP models endpoint failed to respond.")
      logging.debug("LiteRTConnectionStrategy __aenter__: Health check SUCCESS")

      # Non-blocking warm-up completions request (proxy-less urlopen)
      logging.debug(
          "LiteRTConnectionStrategy __aenter__: Starting warm-up request"
      )
      warmup_timeout = float(_WARMUP_REQUEST_TIMEOUT_SECONDS)
      if self._max_kv_cache_tokens:
        warmup_timeout = max(
            warmup_timeout, float(self._max_kv_cache_tokens) / 250.0
        )

      try:
        warmup_payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self._openai_server_url}/v1/chat/completions",
            data=json.dumps(warmup_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        def _warmup():
          logging.debug(
              "LiteRTConnectionStrategy __aenter__: _warmup query start"
          )
          with _urlopen_no_proxy(req, timeout=warmup_timeout) as r:
            r.read()
          logging.debug(
              "LiteRTConnectionStrategy __aenter__: _warmup query complete"
          )

        await loop.run_in_executor(None, _warmup)
        logging.debug(
            "LiteRTConnectionStrategy __aenter__: Warm-up query completed"
            " successfully"
        )
      # pylint: disable=broad-exception-caught
      except Exception as e:  # pylint: disable=broad-exception-caught  # pylint: disable=broad-exception-caught
        logging.warning("LiteRT warm-up request timed out or failed: %s", e)
      finally:
        # Ensure mid-stream warm-up handlers finish and release the engine
        # lock before accepting real user requests from localharness.
        if self._openai_server and hasattr(self._openai_server, "engine_lock"):

          def _wait_for_engine():
            with self._openai_server.engine_lock:
              pass

          await loop.run_in_executor(None, _wait_for_engine)

      # Start Go localharness Subprocess via parent
      await super().__aenter__()

    # pylint: disable=broad-exception-caught
    except BaseException:
      try:
        self._shutdown_server()
      finally:
        self._close_engine()
      raise

  def _shutdown_server(self) -> None:
    """Safely shuts down the loopback server thread."""
    if self._openai_server:
      logging.info("Shutting down LiteRT loopback API server thread")
      try:
        self._openai_server.shutdown()
        self._openai_server.server_close()
        if self._openai_server_thread is not None:
          self._openai_server_thread.join()
          self._openai_server_thread = None
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error during loopback API server shutdown: %s", e)
      finally:
        with self._openai_server.engine_lock:
          self._openai_server.close_conversation()
        self._openai_server = None

  def _close_engine(self) -> None:
    """Safely exits LiteRT engine under engine_lock to prevent race conditions with active handler threads."""
    if self._engine is not None:
      try:
        lock = (
            getattr(self._openai_server, "engine_lock", None)
            if self._openai_server
            else None
        )
        if lock is not None:
          with lock:
            self._engine.__exit__(None, None, None)
        else:
          self._engine.__exit__(None, None, None)
      # pylint: disable=broad-exception-caught
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error exiting LiteRT engine: %s", e)
      finally:
        self._engine_context = None
        self._engine = None

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Exits connection strategy and cleans up loopback server and model engine."""
    try:
      await super().__aexit__(exc_type, exc_val, exc_tb)
    finally:
      try:
        self._shutdown_server()
      finally:
        self._close_engine()


def _urlopen_no_proxy(url_or_req: Any, timeout: float | None = None) -> Any:
  """Executes urllib request bypassing D-Bus and OS proxy auto-detection hooks."""
  url_str = (
      url_or_req.full_url
      if hasattr(url_or_req, "full_url")
      else str(url_or_req)
  )
  logging.debug("urlopen_no_proxy START: %s (timeout=%s)", url_str, timeout)
  try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    res = opener.open(url_or_req, timeout=timeout)
    logging.debug("urlopen_no_proxy SUCCESS: %s", url_str)
    return res
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("urlopen_no_proxy ERROR: %s -> %s", url_str, e)
    raise


def _check_gpu_acceleration_available() -> bool:
  """Verifies GPU acceleration hardware (Metal or CUDA) availability."""
  if sys.platform == "darwin":
    try:
      # pylint: disable=g-import-not-at-top
      import subprocess

      out = subprocess.check_output(
          ["sysctl", "-n", "hw.optional.arm64"]
      ).strip()
      if out == b"1":
        return True
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  elif sys.platform.startswith("linux") or sys.platform == "win32":
    if shutil.which("nvidia-smi") is not None:
      try:
        # pylint: disable=g-import-not-at-top
        import subprocess

        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    # Soft check for loaded graphics drivers DLLs
    try:
      # pylint: disable=g-import-not-at-top
      import ctypes

      if sys.platform == "win32":
        ctypes.windll.LoadLibrary("nvcuda.dll")
        return True
      else:
        try:
          ctypes.CDLL("libcuda.so")
          return True
        except Exception:  # pylint: disable=broad-exception-caught
          ctypes.CDLL("libcuda.so.1")
          return True
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return False
