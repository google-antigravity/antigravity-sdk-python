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

"""GAOS Interactions API connection and strategy for localharness."""

import json
import subprocess
from typing import Any, Sequence

from typing_extensions import override

from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.connections.local import event_processor
from google.antigravity.connections.local import interactions_config_converter
from google.antigravity.connections.local import interactions_event_processor
from google.antigravity.connections.local import local_connection
from google.antigravity.hooks import hook_runner as h_runner
from google.antigravity.hooks import policy
from google.antigravity.tools import tool_runner as t_runner


class InteractionsConnection(local_connection.LocalConnection):
  """Connection to the Go-based local harness using the GAOS Interactions API."""

  def __init__(
      self,
      process: subprocess.Popen[bytes] | None,
      ws: Any,
      root_interaction_id: str = "",
      tool_runner: t_runner.ToolRunner | None = None,
      hook_runner: h_runner.HookRunner | None = None,
      initial_history: Sequence[types.Step] | None = None,
      env: dict[str, str] | None = None,
      debug_config: connection.DebugConfig | None = None,
      dynamic_policy_map: dict[str, policy.Policy] | None = None,
      initial_usage: types.UsageMetadata | None = None,
      initial_trajectory_usages: dict[str, types.UsageMetadata] | None = None,
      sandbox_status: types.SandboxStatus | None = None,
  ):
    self._root_interaction_id = root_interaction_id
    super().__init__(
        process=process,
        ws=ws,
        tool_runner=tool_runner,
        hook_runner=hook_runner,
        initial_history=initial_history,
        env=env,
        debug_config=debug_config,
        dynamic_policy_map=dynamic_policy_map,
        initial_usage=initial_usage,
        initial_trajectory_usages=initial_trajectory_usages,
        sandbox_status=sandbox_status,
    )

  @override
  def _create_event_processor(
      self,
      *,
      hook_runner: h_runner.HookRunner | None,
      tool_runner: t_runner.ToolRunner | None,
      dynamic_policy_map: dict[str, policy.Policy] | None,
      initial_usage: types.UsageMetadata | None,
      initial_trajectory_usages: dict[str, types.UsageMetadata] | None,
  ) -> event_processor.BaseLocalEventProcessor:
    """Creates an InteractionsEventProcessor for this connection."""
    return interactions_event_processor.InteractionsEventProcessor(
        send_json_event_fn=self._send_json_event,
        root_interaction_id=self._root_interaction_id,
        hook_runner=hook_runner,
        tool_runner=tool_runner,
        dynamic_policy_map=dynamic_policy_map,
        initial_usage=initial_usage,
        initial_trajectory_usages=initial_trajectory_usages,
    )

  async def _send_json_event(self, event: dict[str, Any]) -> None:
    """Sends a GAOS JSON ExternalClientEvent dict over the WebSocket."""
    await self._ws.send(json.dumps(event))

  @override
  async def send(self, prompt: types.Content | None, **kwargs: Any) -> None:
    """Sends a prompt to the agent over the GAOS Interactions protocol."""
    self._client_cancelled = False
    self._processor.reset_for_turn()
    effective_prompt: types.Content = "" if prompt is None else prompt
    event = interactions_config_converter.content_to_user_input_event(
        effective_prompt
    )
    await self._send_json_event(event)

  @override
  async def cancel(self) -> None:
    """Cancels the current turn over the GAOS Interactions protocol."""
    self._client_cancelled = True
    await self._send_json_event(
        interactions_config_converter.build_cancel_interaction_event()
    )

  @override
  async def send_trigger_notification(self, content: str) -> None:
    """Raises NotImplementedError because automated triggers are not yet supported."""
    raise NotImplementedError(
        "Automated trigger notifications are not yet supported on"
        " InteractionsConnection."
    )

  @override
  async def _send_session_end_request(self) -> None:
    """Sends an interaction.complete event before disconnecting."""
    await self._send_json_event(
        interactions_config_converter.build_complete_interaction_event()
    )

  @override
  async def _parse_and_process_ws_message(self, raw_msg: str | bytes) -> None:
    """Parses a raw GAOS JSON WebSocket frame and routes it to the processor."""
    event_dict = json.loads(raw_msg)
    assert isinstance(
        self._processor,
        interactions_event_processor.InteractionsEventProcessor,
    )
    await self._processor.process_event(event_dict)


class InteractionsConnectionStrategy(local_connection.LocalConnectionStrategy):
  """Strategy for establishing an InteractionsConnection."""

  def _build_create_interaction_event(self) -> dict[str, Any]:
    """Translates SDK config directly into a GAOS JSON interaction.create event."""
    return interactions_config_converter.build_create_interaction_event(
        models=self._models,
        system_instructions=self._system_instructions,
        capabilities_config=self._capabilities_config,
        compaction_config=self._compaction_config,
        conversation_id=self._conversation_id,
        session_continuation_mode=self._session_continuation_mode,
        workspaces=self._workspaces,
        skills_paths=self._skills_paths,
        app_data_dir=self._default_app_data_dir(),
        mcp_servers=self._mcp_servers,
        subagents=self._subagents,
        retry_config=self._retry_config,
        budget_config=self._budget_config,
        policies=self._policies,
        tools=self._tools,
        tool_runner=self._tool_runner,
        hook_runner=self._hook_runner,
    )

  @override
  async def __aenter__(self) -> None:
    """Spawns localharness in Interactions mode and completes the handshake."""
    self._validate_connection()

    create_event = self._build_create_interaction_event()
    process, ws, ws_url = await self._spawn_harness_and_connect_ws(
        use_interactions_api=True
    )
    assert process.stderr is not None

    root_interaction_id = ""
    try:
      await ws.send(json.dumps(create_event))
      raw_init_resp = await ws.recv()
      if isinstance(raw_init_resp, (str, bytes)):
        init_resp = json.loads(raw_init_resp)
        if init_resp.get("event_type") != "interaction.created":
          raise RuntimeError(
              "Expected interaction.created handshake response, got:"
              f" {init_resp}"
          )
        root_interaction_id = init_resp.get("interaction_id") or ""
    except Exception as e:
      stderr_output = await self._cleanup_failed_handshake(
          process, ws, read_stderr=True
      )
      raise RuntimeError(
          f"Failed to initialize interaction at {ws_url}. Stderr:"
          f" {stderr_output}"
      ) from e
    except BaseException:
      await self._cleanup_failed_handshake(process, ws)
      raise

    self._connection = InteractionsConnection(
        process=process,
        ws=ws,
        root_interaction_id=root_interaction_id,
        tool_runner=self._tool_runner,
        hook_runner=self._hook_runner,
        env=self._env,
        debug_config=self._debug_config,
        dynamic_policy_map=self._dynamic_policy_map or None,
    )
    self._connection._start_stderr_reader(process.stderr)
