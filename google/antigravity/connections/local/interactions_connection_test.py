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

"""Unit and integration tests for interactions_connection."""

import asyncio
import json
import unittest
from unittest import mock

from absl.testing import absltest

from google.antigravity import types
from google.antigravity.connections.local import interactions_connection
from google.antigravity.connections.local import local_connection_config
from google.antigravity.connections.local import test_utils
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy


class InteractionsConnectionUnitTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)
    self.mock_process = mock.MagicMock()
    self.ws = test_utils.TestWebSocket()

  async def test_send_cancel_and_disconnect_with_session_end_hook(self):
    ended = asyncio.Event()

    @hooks.on_session_end
    async def handle_end():
      ended.set()

    hr = hook_runner.HookRunner(on_session_end_hooks=[handle_end])

    conn = interactions_connection.InteractionsConnection(
        process=self.mock_process,
        ws=self.ws,
        root_interaction_id="conv-root",
        hook_runner=hr,
    )
    self.assertEqual(conn.conversation_id, "conv-root")

    # 1. send()
    await conn.send("Hello from Interactions")
    raw_sent = await asyncio.wait_for(self.ws.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        json.loads(raw_sent),
        {
            "event_type": "input",
            "content": [{"type": "text", "text": "Hello from Interactions"}],
        },
    )

    # 2. cancel()
    await conn.cancel()
    raw_cancel = await asyncio.wait_for(self.ws.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        json.loads(raw_cancel), {"event_type": "interaction.cancel"}
    )

    # 3. disconnect() with OnSessionEnd hook
    disconnect_task = asyncio.create_task(conn.disconnect())
    raw_complete = await asyncio.wait_for(self.ws.sent_queue.get(), timeout=2.0)
    self.assertEqual(
        json.loads(raw_complete), {"event_type": "interaction.complete"}
    )

    # Simulate harness sending call_hook_request(on_session_end) then
    # interaction.completed.
    await self.ws.queue.put(
        json.dumps({
            "event_type": "call_hook_request",
            "request_id": "req-end-1",
            "type": "on_session_end",
            "name": "OnSessionEnd",
        })
    )
    raw_hook_resp = await asyncio.wait_for(
        self.ws.sent_queue.get(), timeout=2.0
    )
    self.assertEqual(
        json.loads(raw_hook_resp),
        {
            "event_type": "call_hook_response",
            "request_id": "req-end-1",
            "empty_result": {},
        },
    )
    await self.ws.queue.put(json.dumps({"event_type": "interaction.completed"}))
    await asyncio.wait_for(disconnect_task, timeout=2.0)
    self.assertTrue(ended.is_set())

  def test_interactions_agent_config_creates_interactions_strategy(self):
    cfg = local_connection_config.InteractionsAgentConfig(
        api_key="test-key",
        system_instructions="You are helpful.",
    )
    strategy = cfg.create_strategy(tool_runner=None, hook_runner=None)
    self.assertIsInstance(
        strategy, interactions_connection.InteractionsConnectionStrategy
    )
    create_ev = strategy._build_create_interaction_event()
    self.assertEqual(create_ev["event_type"], "interaction.create")
    self.assertEqual(create_ev["agent"], "antigravity")
    self.assertEqual(create_ev["agent_config"]["type"], "antigravity")

  def test_rejects_auto_and_dynamic_policies(self):
    for unsupported_policy in (
        policy.auto(),
        policy.allow("run_command", when=lambda args: True),
    ):
      cfg = local_connection_config.InteractionsAgentConfig(
          api_key="test-key",
          policies=[unsupported_policy],
      )
      strategy = cfg.create_strategy(tool_runner=None, hook_runner=None)
      with self.assertRaises(types.AntigravityValidationError):
        strategy._build_create_interaction_event()

  async def test_aenter_handshake_failure_kills_process(self):
    cfg = local_connection_config.InteractionsAgentConfig(
        api_key="test-key",
    )
    strategy = cfg.create_strategy(tool_runner=None, hook_runner=None)
    self.mock_process.stderr.read.return_value = b"fatal handshake error"
    self.ws.close = mock.AsyncMock(wraps=self.ws.close)
    await self.ws.queue.put(
        json.dumps({"event_type": "error", "error": {"message": "bad config"}})
    )
    with mock.patch.object(
        strategy,
        "_spawn_harness_and_connect_ws",
        new=mock.AsyncMock(
            return_value=(self.mock_process, self.ws, "ws://127.0.0.1:9999")
        ),
    ):
      with self.assertRaisesRegex(RuntimeError, "fatal handshake error"):
        await strategy.__aenter__()
    self.ws.close.assert_awaited_once()
    self.mock_process.kill.assert_called_once()
    self.mock_process.wait.assert_called_once_with(timeout=1)

  async def test_aenter_cancelled_error_kills_process_and_closes_ws(self):
    cfg = local_connection_config.InteractionsAgentConfig(
        api_key="test-key",
    )
    strategy = cfg.create_strategy(tool_runner=None, hook_runner=None)
    self.ws.close = mock.AsyncMock(wraps=self.ws.close)
    self.ws.recv = mock.AsyncMock(side_effect=asyncio.CancelledError())
    with mock.patch.object(
        strategy,
        "_spawn_harness_and_connect_ws",
        new=mock.AsyncMock(
            return_value=(self.mock_process, self.ws, "ws://127.0.0.1:9999")
        ),
    ):
      with self.assertRaises(asyncio.CancelledError):
        await strategy.__aenter__()
    self.ws.close.assert_awaited_once()
    self.mock_process.kill.assert_called_once()
    self.mock_process.wait.assert_called_once_with(timeout=1)

  async def test_rejects_triggers(self):
    from google.antigravity.triggers import triggers  # pylint: disable=g-import-not-at-top

    @triggers.trigger
    async def my_trigger(ctx: triggers.TriggerContext) -> None:
      await ctx.send("ping")

    with self.assertRaises(types.AntigravityValidationError):
      local_connection_config.InteractionsAgentConfig(
          api_key="test-key",
          triggers=[my_trigger],
      )

    conn = interactions_connection.InteractionsConnection(
        process=self.mock_process,
        ws=self.ws,
        root_interaction_id="conv-root",
    )
    with self.assertRaises(NotImplementedError):
      await conn.send_trigger_notification("ping")

  async def test_usage_update_accumulates_root_and_subagent_usage(self):
    conn = interactions_connection.InteractionsConnection(
        process=self.mock_process,
        ws=self.ws,
        root_interaction_id="conv-root",
    )
    conn._is_idle.clear()
    # 1. Root trajectory usage_update (proto3 implicit presence omits zeros)
    await self.ws.queue.put(
        json.dumps({
            "event_type": "state_update",
            "state": "running",
        })
    )
    await self.ws.queue.put(
        json.dumps({
            "event_type": "usage_update",
            "total_input_tokens": 100,
            "total_output_tokens": 25,
            "total_tokens": 125,
        })
    )
    # 1b. Second root trajectory usage_update (nested usage_update payload)
    await self.ws.queue.put(
        json.dumps({
            "event_type": "usage_update",
            "usage_update": {
                "total_input_tokens": 50,
                "total_output_tokens": 10,
                "total_tokens": 60,
            },
        })
    )
    # 2. Subagent trajectory usage_update
    await self.ws.queue.put(
        json.dumps({
            "event_type": "usage_update",
            "sub_interaction_id": "sub-conv-1",
            "total_input_tokens": 40,
            "total_cached_tokens": 10,
            "total_output_tokens": 15,
            "total_thought_tokens": 5,
            "total_tokens": 70,
        })
    )
    # 3. Complete turn via state_update(fully_idle) so receive_steps drains
    await self.ws.queue.put(
        json.dumps({
            "event_type": "state_update",
            "state": "fully_idle",
        })
    )
    async for _ in conn.receive_steps():
      pass

    self.assertEqual(
        conn.cumulative_usage,
        types.UsageMetadata(
            prompt_token_count=190,
            cached_content_token_count=10,
            candidates_token_count=50,
            thoughts_token_count=5,
            total_token_count=255,
        ),
    )
    self.assertEqual(
        conn.trajectory_usages["conv-root"],
        types.UsageMetadata(
            prompt_token_count=150,
            cached_content_token_count=0,
            candidates_token_count=35,
            thoughts_token_count=0,
            total_token_count=185,
        ),
    )
    self.assertEqual(
        conn.trajectory_usages["sub-conv-1"],
        types.UsageMetadata(
            prompt_token_count=40,
            cached_content_token_count=10,
            candidates_token_count=15,
            thoughts_token_count=5,
            total_token_count=70,
        ),
    )
    await conn.disconnect()

  async def test_tool_hooks_correlate_call_id_and_step_id(self):
    pre_calls: list[types.ToolCall] = []
    post_results: list[types.ToolResult] = []
    tool_errors: list[types.ToolExecutionError] = []

    @hooks.pre_tool_call_decide
    async def on_pre_tool(data: types.ToolCall) -> types.HookResult:
      pre_calls.append(data)
      return types.HookResult(allow=True)

    @hooks.post_tool_call
    async def on_post_tool(data: types.ToolResult) -> None:
      post_results.append(data)

    @hooks.on_tool_error
    async def on_tool_err(data: types.ToolExecutionError) -> None:
      tool_errors.append(data)

    hr = hook_runner.HookRunner(
        pre_tool_call_decide_hooks=[on_pre_tool],
        post_tool_call_hooks=[on_post_tool],
        on_tool_error_hooks=[on_tool_err],
    )
    conn = interactions_connection.InteractionsConnection(
        process=self.mock_process,
        ws=self.ws,
        hook_runner=hr,
        root_interaction_id="conv-root",
    )
    conn._is_idle.clear()
    await self.ws.queue.put(
        json.dumps({"event_type": "state_update", "state": "running"})
    )

    # 1. User input step at index 0
    await self.ws.queue.put(
        json.dumps({
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "user_input", "content": "Hello"},
        })
    )
    await self.ws.queue.put(json.dumps({"event_type": "step.stop", "index": 0}))

    # 2. PreTool hook fires with explicit call_id
    await self.ws.queue.put(
        json.dumps({
            "event_type": "call_hook_request",
            "request_id": "req-pre-1",
            "type": "pre_tool",
            "name": "PreTool",
            "pre_tool_args": {
                "tool_name": "get_weather",
                "arguments_json": '{"location": "Seattle"}',
                "call_id": "call-weather-1",
            },
        })
    )
    await asyncio.wait_for(self.ws.sent_queue.get(), timeout=2.0)

    # 3. step.start(function_call) at index 1 with id="call-weather-1"
    await self.ws.queue.put(
        json.dumps({
            "event_type": "step.start",
            "index": 1,
            "step": {
                "type": "function_call",
                "id": "call-weather-1",
                "name": "get_weather",
                "arguments": {"location": "Seattle"},
            },
        })
    )
    await self.ws.queue.put(json.dumps({"event_type": "step.stop", "index": 1}))

    # 4. PostTool hook fires with explicit call_id
    await self.ws.queue.put(
        json.dumps({
            "event_type": "call_hook_request",
            "request_id": "req-post-1",
            "type": "post_tool",
            "name": "PostTool",
            "post_tool_args": {
                "tool_name": "get_weather",
                "result": "Sunny",
                "call_id": "call-weather-1",
            },
        })
    )
    await asyncio.wait_for(self.ws.sent_queue.get(), timeout=2.0)

    # 5. Built-in tool error at index 2 followed by OnToolError hook
    await self.ws.queue.put(
        json.dumps({
            "event_type": "step.start",
            "index": 2,
            "step": {
                "type": "code_execution_call",
                "id": "call-cmd-2",
                "code": "false",
            },
        })
    )
    await self.ws.queue.put(json.dumps({"event_type": "step.stop", "index": 2}))
    await self.ws.queue.put(
        json.dumps({
            "event_type": "step.start",
            "index": 3,
            "step": {
                "type": "code_execution_result",
                "call_id": "call-cmd-2",
                "is_error": True,
                "result": "exit status 1",
            },
        })
    )
    await self.ws.queue.put(json.dumps({"event_type": "step.stop", "index": 3}))
    await self.ws.queue.put(
        json.dumps({
            "event_type": "call_hook_request",
            "request_id": "req-err-1",
            "type": "on_tool_error",
            "name": "OnToolError",
            "on_tool_error_args": {
                "tool_name": "run_command",
                "error_message": "exit status 1",
                "call_id": "call-cmd-2",
            },
        })
    )
    await asyncio.wait_for(self.ws.sent_queue.get(), timeout=2.0)

    await self.ws.queue.put(
        json.dumps({"event_type": "state_update", "state": "fully_idle"})
    )
    emitted_steps = [step async for step in conn.receive_steps()]
    tool_steps = [
        s for s in emitted_steps if s.type == types.StepType.TOOL_CALL
    ]
    self.assertEqual(len(tool_steps), 3)
    self.assertEqual(tool_steps[0].tool_calls[0].id, "call-weather-1")
    self.assertEqual(tool_steps[0].tool_calls[0].step_id, "call-weather-1")
    self.assertEqual(tool_steps[1].tool_calls[0].id, "call-cmd-2")
    self.assertEqual(tool_steps[1].tool_calls[0].step_id, "call-cmd-2")
    self.assertEqual(tool_steps[2].tool_calls[0].id, "call-cmd-2")
    self.assertEqual(tool_steps[2].tool_calls[0].step_id, "call-cmd-2")

    self.assertEqual(len(pre_calls), 1)
    self.assertEqual(pre_calls[0].id, "call-weather-1")
    self.assertEqual(pre_calls[0].step_id, "call-weather-1")
    self.assertEqual(pre_calls[0].id, tool_steps[0].tool_calls[0].id)
    self.assertEqual(pre_calls[0].step_id, tool_steps[0].tool_calls[0].step_id)

    self.assertEqual(len(post_results), 1)
    self.assertEqual(post_results[0].id, "call-weather-1")
    self.assertEqual(post_results[0].step_id, "call-weather-1")
    self.assertEqual(post_results[0].id, tool_steps[0].tool_calls[0].id)
    self.assertEqual(
        post_results[0].step_id, tool_steps[0].tool_calls[0].step_id
    )

    self.assertEqual(len(tool_errors), 1)
    self.assertEqual(tool_errors[0].call_id, "call-cmd-2")
    self.assertEqual(tool_errors[0].step_id, "call-cmd-2")
    self.assertEqual(tool_errors[0].call_id, tool_steps[2].tool_calls[0].id)
    self.assertEqual(
        tool_errors[0].step_id, tool_steps[2].tool_calls[0].step_id
    )
    await conn.disconnect()


if __name__ == "__main__":
  absltest.main()
