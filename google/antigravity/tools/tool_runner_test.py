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

"""Tests for tool_runner module."""

import asyncio
import dataclasses
import threading
from typing import Optional, Union
from unittest import mock

from absl.testing import absltest
import pydantic

from google.antigravity import types as sdk_types
from google.antigravity.tools import tool_runner


def _sample_tool(arg1: str) -> str:
  return f"Hello {arg1}"


async def _async_tool(x: int, y: int) -> int:
  return x + y


class _CustomData(pydantic.BaseModel):
  value: int


def _tool_with_string_ann(data: "_CustomData") -> int:
  return data.value


class ToolRunnerTest(absltest.TestCase):
  """Validates the in-process ToolRunner.

  Ensures that Python tools can be registered, unregistered, listed,
  and executed correctly, handling both sync and async callables.
  """

  def test_register_and_list(self):
    """Verifies tool registration and listing.

    What: Checks that a tool registered without explicit name uses __name__.
    Why: Validates default naming behavior and listing completeness.
    How: Registers a sample tool and asserts its name is in tool_names.
    """

    runner = tool_runner.ToolRunner()
    runner.register(_sample_tool)
    self.assertEqual(runner.tool_names, ["_sample_tool"])

  def test_register_with_custom_name(self):
    """Verifies registration with a custom name override.

    What: Checks that a tool registered with an explicit name uses that name.
    Why: Validates that users can alias tools or avoid naming collisions.
    How: Registers a tool with name="greet" and asserts "greet" is listed.
    """

    runner = tool_runner.ToolRunner()
    runner.register(_sample_tool, name="greet")
    self.assertEqual(runner.tool_names, ["greet"])

  def test_register_duplicate_raises(self):
    """Verifies that duplicate tool registration is forbidden.

    What: Checks that duplicate registration fails with ValueError.
    Why: Prevents accidental overwriting of tools.
    How: Attempts double registration within assertRaises.
    """

    runner = tool_runner.ToolRunner()
    runner.register(_sample_tool)
    with self.assertRaises(ValueError):
      runner.register(_sample_tool)

  def test_unregister(self):
    """Verifies tool removal.

    What: Checks that registered tools can be removed.
    Why: Supports dynamic registry management.
    How: Registers a tool, removes it, and asserts it is no longer listed.
    """

    runner = tool_runner.ToolRunner()
    runner.register(_sample_tool)
    runner.unregister("_sample_tool")
    self.assertEqual(runner.tool_names, [])

  def test_unregister_missing_raises(self):
    """Verifies that removing a non-existent tool raises KeyError.

    What: Checks error behavior for invalid unregister requests.
    Why: Confirms that removing a missing tool is an error.
    How: Calls unregister for a placeholder name within assertRaises.
    """

    runner = tool_runner.ToolRunner()
    with self.assertRaises(KeyError):
      runner.unregister("nonexistent")

  def test_execute_sync_tool(self):
    """Verifies execution of standard synchronous tools.

    What: Checks that execution invokes the callable and returns its value.
    Why: Validates basic sync tool execution path.
    How: Executes a sync placeholder tool and asserts the return message.
    """

    runner = tool_runner.ToolRunner([_sample_tool])
    result = asyncio.run(runner.execute("_sample_tool", arg1="World"))
    self.assertEqual(result, "Hello World")

  def test_execute_sync_tool_in_thread(self):
    """Verifies that sync tools are executed in a separate thread.

    Why: If a tool is executed within the even loop, then it must not do
    blocking operations, which is not realistic.
    """

    main_thread_id = threading.get_ident()
    tool_thread_id = None

    def _thread_check_tool():
      nonlocal tool_thread_id
      tool_thread_id = threading.get_ident()
      return "ok"

    runner = tool_runner.ToolRunner([_thread_check_tool])
    result = asyncio.run(runner.execute("_thread_check_tool"))
    self.assertEqual(result, "ok")
    self.assertNotEqual(main_thread_id, tool_thread_id)

  def test_execute_async_tool(self):
    """Verifies execution of asynchronous (coroutine) tools.

    What: Checks that execution awaits the coroutine and returns its value.
    Why: Validates async tool execution path.
    How: Executes an async placeholder tool and asserts the return sum.
    """

    runner = tool_runner.ToolRunner([_async_tool])
    result = asyncio.run(runner.execute("_async_tool", x=3, y=4))
    self.assertEqual(result, 7)

  def test_execute_unknown_tool_raises(self):
    """Verifies that executing an unregistered tool raises KeyError.

    What: Checks error behavior for invalid execution requests.
    Why: Alerts caller that requested tool is missing.
    How: Invokes execute with a placeholder name within assertRaises.
    """

    runner = tool_runner.ToolRunner()
    with self.assertRaises(KeyError):
      asyncio.run(runner.execute("nonexistent"))

  def test_init_with_tools_list(self):
    """Verifies constructor-based tool registration.

    What: Checks that tools provided during init are registered.
    Why: Allows bulk registration on startup.
    How: Inits ToolRunner with two tools and asserts both are listed.
    """

    runner = tool_runner.ToolRunner([_sample_tool, _async_tool])
    self.assertLen(runner.tool_names, 2)
    self.assertIn("_sample_tool", runner.tool_names)
    self.assertIn("_async_tool", runner.tool_names)

  def test_execute_tool_failure_raises_exception(self):
    """Verifies that tool internal crashes are propagated as exceptions."""

    def _failing_tool():
      raise ValueError("Something went wrong")

    runner = tool_runner.ToolRunner([_failing_tool])
    with self.assertRaises(ValueError) as cm:
      asyncio.run(runner.execute("_failing_tool"))
    self.assertEqual(str(cm.exception), "Something went wrong")

  def test_tool_with_schema_sync(self):
    """Verifies ToolWithSchema with a synchronous callable.

    What: Checks that ToolWithSchema wrapper can wrap synchronous callables and
    be executed safely by the ToolRunner.
    Why: Covers manual wrapping use-cases where users need explicit schemas
    attached to synchronous methods.
    How: Registers a wrapped synchronous placeholder tool, executes it, and asserts
    expected return string.
    """
    tool = tool_runner.ToolWithSchema(_sample_tool, {"type": "object"})
    runner = tool_runner.ToolRunner([tool])
    result = asyncio.run(runner.execute("_sample_tool", arg1="World"))
    self.assertEqual(result, "Hello World")

  def test_tool_with_schema_async(self):
    """Verifies ToolWithSchema with an asynchronous callable.

    What: Checks that ToolWithSchema wrapper can wrap asynchronous callables and
    be executed safely by the ToolRunner.
    Why: Covers manual wrapping use-cases where users need explicit schemas
    attached to asynchronous methods (e.g. MCP tools).
    How: Registers a wrapped asynchronous placeholder tool, executes it, and asserts
    expected return sum.
    """
    tool = tool_runner.ToolWithSchema(_async_tool, {"type": "object"})
    runner = tool_runner.ToolRunner([tool])
    result = asyncio.run(runner.execute("_async_tool", x=3, y=4))
    self.assertEqual(result, 7)

  def test_tool_with_schema_async_does_not_use_thread(self):
    """Verifies that ToolWithSchema with async callable does not use to_thread."""
    with mock.patch(
        "asyncio.to_thread", wraps=asyncio.to_thread
    ) as mock_to_thread:
      tool = tool_runner.ToolWithSchema(_async_tool, {"type": "object"})
      runner = tool_runner.ToolRunner([tool])
      result = asyncio.run(runner.execute("_async_tool", x=3, y=4))
      self.assertEqual(result, 7)
      mock_to_thread.assert_not_called()

  def test_tools_property(self):
    """Verifies tools property returns a copy of the dictionary."""
    runner = tool_runner.ToolRunner([_sample_tool])
    tools_dict = runner.tools
    self.assertEqual(tools_dict, {"_sample_tool": _sample_tool})
    # Verify it's a copy
    tools_dict["other"] = _async_tool
    self.assertNotIn("other", runner.tools)

  def test_execute_sync_callable_object(self):
    """Verifies execution of a class instance with a sync __call__ method."""

    class SyncCallable:

      def __call__(self, arg1: str) -> str:
        return f"Callable {arg1}"

    tool = SyncCallable()
    runner = tool_runner.ToolRunner()
    runner.register(tool, name="sync_callable")

    result = asyncio.run(runner.execute("sync_callable", arg1="World"))
    self.assertEqual(result, "Callable World")

  def test_register_callable_object_no_name(self):
    """Verifies registering a callable object without name uses class name."""

    class SyncCallable:

      def __call__(self, arg1: str) -> str:
        return f"Callable {arg1}"

    tool = SyncCallable()
    runner = tool_runner.ToolRunner()
    runner.register(tool)
    self.assertEqual(runner.tool_names, ["SyncCallable"])

  def test_tool_with_schema_no_name(self):
    """Verifies ToolWithSchema handles callable object without __name__."""

    class SyncCallable:

      def __call__(self, arg1: str) -> str:
        return f"Callable {arg1}"

    tool = SyncCallable()
    wrapped = tool_runner.ToolWithSchema(tool, {"type": "object"})
    self.assertEqual(wrapped.__name__, "SyncCallable")

  def test_coerce_args_basic_types(self):
    """Verifies that _coerce_args converts strings to basic Python types."""

    def _typed_fn(a: int, b: float, c: bool, d: str) -> None:
      del a, b, c, d

    runner = tool_runner.ToolRunner([_typed_fn])
    coerced = runner._coerce_args(
        _typed_fn, {"a": "5", "b": "2.5", "c": "true", "d": "hello"}
    )
    self.assertEqual(coerced, {"a": 5, "b": 2.5, "c": True, "d": "hello"})
    self.assertIsInstance(coerced["a"], int)
    self.assertIsInstance(coerced["b"], float)
    self.assertIsInstance(coerced["c"], bool)
    self.assertIsInstance(coerced["d"], str)

  def test_coerce_args_string_annotations(self):
    """Verifies that _coerce_args handles string annotations (forward refs)."""
    runner = tool_runner.ToolRunner([_tool_with_string_ann])
    coerced = runner._coerce_args(
        _tool_with_string_ann, {"data": {"value": 42}}
    )
    self.assertIn("data", coerced)
    self.assertIsInstance(coerced["data"], _CustomData)
    self.assertEqual(coerced["data"].value, 42)

  def test_coerce_args_optional_and_union(self):
    """Verifies that _coerce_args handles Optional and Union types."""

    def _complex_fn(x: Optional[int], y: Union[int, float]) -> None:
      del x, y

    runner = tool_runner.ToolRunner([_complex_fn])
    coerced = runner._coerce_args(_complex_fn, {"x": "10", "y": "3.14"})
    self.assertEqual(coerced, {"x": 10, "y": 3.14})
    self.assertIsInstance(coerced["x"], int)
    self.assertIsInstance(coerced["y"], float)

    coerced_none = runner._coerce_args(_complex_fn, {"x": None, "y": 42})
    self.assertEqual(coerced_none, {"x": None, "y": 42})

  def test_coerce_args_fallback_on_error(self):
    """Verifies that _coerce_args retains original values when conversion fails."""

    def _int_fn(count: int) -> None:
      del count

    runner = tool_runner.ToolRunner([_int_fn])
    coerced = runner._coerce_args(_int_fn, {"count": "not_an_int"})
    self.assertEqual(coerced, {"count": "not_an_int"})

  def test_coerce_args_with_tool_with_schema(self):
    """Verifies that _coerce_args unwraps ToolWithSchema."""

    def _typed_tool(val: int) -> int:
      return val * 2

    wrapped = tool_runner.ToolWithSchema(_typed_tool, {"type": "object"})
    runner = tool_runner.ToolRunner([wrapped])
    coerced = runner._coerce_args(wrapped, {"val": "21"})
    self.assertEqual(coerced, {"val": 21})

  def test_execute_coerces_arguments(self):
    """Verifies that execute() automatically coerces argument types."""

    def _sum_tool(x: int, y: float) -> float:
      return x + y

    runner = tool_runner.ToolRunner([_sum_tool])
    result = asyncio.run(runner.execute("_sum_tool", x="10", y="5.5"))
    self.assertEqual(result, 15.5)


class ProcessToolCallsTest(absltest.TestCase):

  """Validates batch tool call processing via process_tool_calls.

  Ensures that normalized tool call dicts are dispatched correctly and
  results are returned as structured ToolResult objects.
  """

  def test_single_tool_call(self):
    """Verifies processing a single tool call.

    What: Checks that a single normalized tool call dict is executed correctly.
    Why: Validates the basic batch processing path.
    How: Processes one call and asserts the ToolResult has the expected value.
    """
    runner = tool_runner.ToolRunner([_async_tool])
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="_async_tool", args={"x": 3, "y": 7})]
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].name, "_async_tool")
    self.assertEqual(results[0].result, 10)
    self.assertIsNone(results[0].error)

  def test_multiple_tool_calls(self):
    """Verifies processing multiple tool calls in a batch.

    What: Checks that all tool calls in the batch are executed in order.
    Why: Validates that batch processing handles multiple items correctly.
    How: Processes two calls and asserts both results.
    """
    runner = tool_runner.ToolRunner([_sample_tool, _async_tool])
    results = asyncio.run(
        runner.process_tool_calls([
            sdk_types.ToolCall(name="_sample_tool", args={"arg1": "World"}),
            sdk_types.ToolCall(name="_async_tool", args={"x": 1, "y": 2}),
        ])
    )
    self.assertLen(results, 2)
    self.assertEqual(results[0].result, "Hello World")
    self.assertEqual(results[1].result, 3)

  def test_unknown_tool_returns_error_result(self):
    """Verifies that unknown tools produce an error ToolResult.

    What: Checks that calling an unregistered tool returns a
    ToolResult with error.
    Why: Unknown tools should not raise; they should report gracefully.
    How: Processes a call to a non-existent tool and asserts the error field.
    """
    runner = tool_runner.ToolRunner()
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="nonexistent", args={})]
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].name, "nonexistent")
    self.assertIsNone(results[0].result)
    self.assertIn("Unknown tool", results[0].error)

  def test_failing_tool_returns_error_result(self):
    """Verifies that a tool that raises produces an error ToolResult.

    What: Checks that internal tool crashes are captured as error ToolResults.
    Why: Prevents a single tool failure from aborting the entire batch.
    How: Processes a call to a crashing tool and asserts the error field.
    """

    def _bad_tool():
      raise RuntimeError("kaboom")

    runner = tool_runner.ToolRunner([_bad_tool])
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="_bad_tool", args={})]
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].name, "_bad_tool")
    self.assertEqual(results[0].error, "kaboom")

  def test_missing_args_defaults_to_empty(self):
    """Verifies that tool calls without 'args' key default to empty dict.

    What: Checks that omitting 'args' doesn't crash.
    Why: Some backends may omit args for zero-argument tools.
    How: Processes a call without 'args' and asserts successful execution.
    """

    def _no_args_tool():
      return "ok"

    runner = tool_runner.ToolRunner([_no_args_tool])
    results = asyncio.run(
        runner.process_tool_calls([sdk_types.ToolCall(name="_no_args_tool")])
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].result, "ok")

  def test_process_tool_calls_with_schema(self):
    """Verifies batch processing of ToolWithSchema.

    What: Checks that ToolWithSchema wrapper is safely executed when batched
    processed by ToolRunner.process_tool_calls.
    Why: Validates batch dispatch mechanisms properly unroll wrapper callables
    safely.
    How: Processes a batch tool call containing a ToolWithSchema wrapped
    coroutine and asserts the result in ToolResult.
    """
    tool = tool_runner.ToolWithSchema(_async_tool, {"type": "object"})
    runner = tool_runner.ToolRunner([tool])
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="_async_tool", args={"x": 3, "y": 7})]
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].result, 10)

  def test_failing_tool_preserves_original_exception(self):
    """Verifies that a tool's original exception is preserved on ToolResult.

    What: Checks that ToolResult.exception holds the original exception object.
    Why: OnToolErrorHook needs the original exception type for isinstance
      dispatch.
    How: Processes a call to a tool that raises ValueError and asserts the
      exception field is the original ValueError instance.
    """

    def _typed_error_tool():
      raise ValueError("bad input")

    runner = tool_runner.ToolRunner([_typed_error_tool])
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="_typed_error_tool", args={})]
        )
    )
    self.assertLen(results, 1)
    self.assertIsNotNone(results[0].exception)
    self.assertIsInstance(results[0].exception, ValueError)
    self.assertIn("bad input", str(results[0].exception))

  def test_mixed_batch_failure_does_not_swallow_successes(self):
    """Verifies that one failing tool does not discard sibling results.

    What: Checks that a batch with both passing and failing tools returns
      all results — successes and errors — without swallowing any.
    Why: With concurrent execution via asyncio.gather, a failing task
      must not cancel siblings or lose their results.
    How: Processes a batch of three tools (success, failure, success) and
      asserts all three ToolResults are present with correct values.
    """

    def _good_tool(x: int) -> int:
      return x * 10

    def _bad_tool():
      raise RuntimeError("kaboom")

    runner = tool_runner.ToolRunner([_good_tool, _bad_tool])
    results = asyncio.run(
        runner.process_tool_calls([
            sdk_types.ToolCall(name="_good_tool", args={"x": 1}),
            sdk_types.ToolCall(name="_bad_tool", args={}),
            sdk_types.ToolCall(name="_good_tool", args={"x": 2}),
        ])
    )
    self.assertLen(results, 3)
    # First call: success.
    self.assertEqual(results[0].result, 10)
    self.assertIsNone(results[0].error)
    # Second call: failure.
    self.assertEqual(results[1].error, "kaboom")
    self.assertIsInstance(results[1].exception, RuntimeError)
    # Third call: success (not swallowed by sibling failure).
    self.assertEqual(results[2].result, 20)
    self.assertIsNone(results[2].error)

  def test_exception_excluded_from_serialization(self):
    """Verifies that ToolResult.exception is not included in model_dump.

    Why: Exception objects are not JSON-serializable; they must be excluded
      from Pydantic serialization while remaining accessible in-memory.
    """
    result = sdk_types.ToolResult(
        name="test",
        error="something broke",
        exception=ValueError("original"),
    )
    dumped = result.model_dump()
    self.assertNotIn("exception", dumped)
    # But the in-memory object still has it.
    self.assertIsInstance(result.exception, ValueError)

  def test_successful_tool_has_no_exception(self):
    """Verifies that successful tool calls have exception=None."""
    runner = tool_runner.ToolRunner([_async_tool])
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="_async_tool", args={"x": 1, "y": 2})]
        )
    )
    self.assertLen(results, 1)
    self.assertIsNone(results[0].exception)

  def test_process_tool_calls_preserves_metadata(self):
    """Verifies that id, step_id, and server_name are preserved on ToolResult."""
    runner = tool_runner.ToolRunner([_sample_tool])
    results = asyncio.run(
        runner.process_tool_calls([
            sdk_types.ToolCall(
                id="call_123",
                step_id="step_456",
                server_name="test_server",
                name="_sample_tool",
                args={"arg1": "Antigravity"},
            )
        ])
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].id, "call_123")
    self.assertEqual(results[0].step_id, "step_456")
    self.assertEqual(results[0].server_name, "test_server")
    self.assertEqual(results[0].result, "Hello Antigravity")

  def test_process_tool_calls_failure_preserves_metadata(self):
    """Verifies that id, step_id, and server_name are preserved on failed ToolResult."""

    def _failing_tool():
      raise RuntimeError("boom")

    runner = tool_runner.ToolRunner([_failing_tool])
    results = asyncio.run(
        runner.process_tool_calls([
            sdk_types.ToolCall(
                id="call_fail",
                step_id="step_fail",
                server_name="fail_server",
                name="_failing_tool",
            )
        ])
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].id, "call_fail")
    self.assertEqual(results[0].step_id, "step_fail")
    self.assertEqual(results[0].server_name, "fail_server")
    self.assertEqual(results[0].error, "boom")
    self.assertIsInstance(results[0].exception, RuntimeError)

  def test_process_tool_calls_unknown_tool_preserves_metadata(self):
    """Verifies that id, step_id, and server_name are preserved on unknown tool result."""
    runner = tool_runner.ToolRunner()
    results = asyncio.run(
        runner.process_tool_calls([
            sdk_types.ToolCall(
                id="call_unknown",
                step_id="step_unknown",
                server_name="unknown_server",
                name="nonexistent_tool",
            )
        ])
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].id, "call_unknown")
    self.assertEqual(results[0].step_id, "step_unknown")
    self.assertEqual(results[0].server_name, "unknown_server")
    self.assertIn("Unknown tool", results[0].error)


class ContextInjectionTest(absltest.TestCase):
  """Validates ToolContext injection into tools.

  Ensures that tools declaring a ToolContext-typed parameter receive the
  context automatically, while tools without it remain unaffected.
  """

  def _make_mock_context(self):
    """Creates a mock ToolContext for testing."""
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    ctx = mock.MagicMock(spec=tool_context.ToolContext)
    ctx.conversation_id = "test-id"
    return ctx

  def test_tool_with_context_receives_it(self):
    """Verifies that a tool requesting ToolContext gets it injected.

    What: Checks that a tool with a ToolContext param receives it.
    Why: Core injection feature — tools must be able to access context.
    How: Registers a tool with a context param, sets context, and executes.
    """
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    received_ctx = None

    def _context_tool(arg1: str, ctx: tool_context.ToolContext) -> str:
      nonlocal received_ctx
      received_ctx = ctx
      return f"got {arg1}"

    mock_ctx = self._make_mock_context()
    runner = tool_runner.ToolRunner([_context_tool])
    runner.set_context(mock_ctx)
    result = asyncio.run(runner.execute("_context_tool", arg1="hello"))
    self.assertEqual(result, "got hello")
    self.assertIs(received_ctx, mock_ctx)

  def test_tool_without_context_works_normally(self):
    """Verifies backward compatibility — tools without context are unaffected.

    What: Checks that plain tools still work when context is set.
    Why: Context injection must be opt-in, not breaking existing tools.
    How: Registers a plain tool, sets context, and verifies normal execution.
    """
    mock_ctx = self._make_mock_context()
    runner = tool_runner.ToolRunner([_sample_tool])
    runner.set_context(mock_ctx)
    result = asyncio.run(runner.execute("_sample_tool", arg1="World"))
    self.assertEqual(result, "Hello World")

  def test_async_tool_with_context(self):
    """Verifies context injection works for async tools.

    What: Checks that async tools with ToolContext get it injected.
    Why: Async tools are first-class citizens and must support injection.
    How: Registers an async tool with context param and verifies injection.
    """
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    received_ctx = None

    async def _async_context_tool(x: int, ctx: tool_context.ToolContext) -> int:
      nonlocal received_ctx
      received_ctx = ctx
      return x * 2

    mock_ctx = self._make_mock_context()
    runner = tool_runner.ToolRunner([_async_context_tool])
    runner.set_context(mock_ctx)
    result = asyncio.run(runner.execute("_async_context_tool", x=5))
    self.assertEqual(result, 10)
    self.assertIs(received_ctx, mock_ctx)

  def test_tool_with_pep604_union_context_receives_it(self):
    """Verifies context injection works with PEP 604 union types (X | None)."""
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    received_ctx = None

    def _pep604_context_tool(
        arg1: str, ctx: tool_context.ToolContext | None
    ) -> str:
      nonlocal received_ctx
      received_ctx = ctx
      return f"got {arg1}"

    mock_ctx = self._make_mock_context()
    runner = tool_runner.ToolRunner([_pep604_context_tool])
    runner.set_context(mock_ctx)
    result = asyncio.run(runner.execute("_pep604_context_tool", arg1="hello"))
    self.assertEqual(result, "got hello")
    self.assertIs(received_ctx, mock_ctx)

  def test_no_context_set_skips_injection(self):
    """Verifies graceful behavior when no context has been set.

    What: Checks that tools with context param don't crash when context is None.
    Why: Context is optional — runner without set_context must still work.
    How: Registers a tool with optional context and runs without set_context.
    """
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    def _optional_ctx_tool(
        arg1: str, ctx: tool_context.ToolContext | None = None
    ) -> str:
      del arg1  # Unused, exists to verify injection doesn't break extra args.
      return f"ctx={ctx is not None}"

    runner = tool_runner.ToolRunner([_optional_ctx_tool])
    # No set_context call — context remains None.
    result = asyncio.run(runner.execute("_optional_ctx_tool", arg1="test"))
    self.assertEqual(result, "ctx=False")

  def test_return_type_not_mistaken_for_param(self):
    """Verifies that a ToolContext return type is not treated as a parameter.

    What: Checks that get_type_hints' 'return' key is skipped.
    Why: get_type_hints includes the return annotation under the 'return' key,
      which could falsely match ToolContext if not filtered.
    How: Defines a tool that returns ToolContext but takes no context param,
      and verifies it works without injection.
    """
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    def _returns_ctx(arg1: str) -> tool_context.ToolContext:
      del arg1
      return None  # type: ignore[return-value]

    runner = tool_runner.ToolRunner([_returns_ctx])
    # Should NOT detect a context param from the return type.
    result = asyncio.run(runner.execute("_returns_ctx", arg1="test"))
    self.assertIsNone(result)

  def test_process_tool_calls_with_context(self):
    """Verifies context injection works in batch processing.

    What: Checks that process_tool_calls injects context correctly.
    Why: Batch processing is the primary tool dispatch path.
    How: Processes a tool call batch and verifies context was injected.
    """
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    received_ctx = None

    def _batch_ctx_tool(ctx: tool_context.ToolContext) -> str:
      nonlocal received_ctx
      received_ctx = ctx
      return "ok"

    mock_ctx = self._make_mock_context()
    runner = tool_runner.ToolRunner([_batch_ctx_tool])
    runner.set_context(mock_ctx)
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="_batch_ctx_tool", args={})]
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].result, "ok")
    self.assertIs(received_ctx, mock_ctx)

  def test_context_not_injected_when_already_in_kwargs(self):
    """Verifies that explicit context kwargs are not overwritten.

    What: Checks that injection does not clobber explicit arguments.
    Why: If a caller explicitly provides the context param, injection
    must respect it to avoid surprising overwrites.
    How: Passes the context param explicitly and verifies it's used.
    """
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    received_ctx = None

    def _ctx_tool(ctx: tool_context.ToolContext) -> str:
      nonlocal received_ctx
      received_ctx = ctx
      return "ok"

    mock_ctx = self._make_mock_context()
    explicit_ctx = self._make_mock_context()
    explicit_ctx.conversation_id = "explicit-id"

    runner = tool_runner.ToolRunner([_ctx_tool])
    runner.set_context(mock_ctx)
    asyncio.run(runner.execute("_ctx_tool", ctx=explicit_ctx))
    self.assertIs(received_ctx, explicit_ctx)

  def test_unregister_cleans_context_param_cache(self):
    """Verifies that unregistering a tool removes its context param cache.

    What: Checks that _context_params is cleaned up on unregister.
    Why: Stale cache entries for removed tools could cause issues on
    re-registration.
    How: Registers and unregisters a context tool, verifies cache is clean.
    """
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    def _ctx_tool(ctx: tool_context.ToolContext) -> str:
      return "ok"

    runner = tool_runner.ToolRunner([_ctx_tool])
    self.assertIn("_ctx_tool", runner._context_params)
    runner.unregister("_ctx_tool")
    self.assertNotIn("_ctx_tool", runner._context_params)

  def test_tool_with_schema_and_context_receives_it(self):
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top
    received_ctx = None

    def _context_tool(arg1: str, ctx: tool_context.ToolContext) -> str:
      nonlocal received_ctx
      received_ctx = ctx
      return f"got {arg1}"

    tool = tool_runner.ToolWithSchema(_context_tool, {"type": "object"})
    mock_ctx = self._make_mock_context()
    runner = tool_runner.ToolRunner([tool])
    runner.set_context(mock_ctx)
    result = asyncio.run(runner.execute("_context_tool", arg1="hello"))
    self.assertEqual(result, "got hello")
    self.assertIs(received_ctx, mock_ctx)

  def test_nested_tool_with_schema_and_context_receives_it(self):
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top
    received_ctx = None

    def _context_tool(arg1: str, ctx: tool_context.ToolContext) -> str:
      nonlocal received_ctx
      received_ctx = ctx
      return f"got {arg1}"

    tool = tool_runner.ToolWithSchema(
        tool_runner.ToolWithSchema(_context_tool, {"type": "inner"}),
        {"type": "outer"},
    )
    mock_ctx = self._make_mock_context()
    runner = tool_runner.ToolRunner([tool])
    runner.set_context(mock_ctx)
    result = asyncio.run(runner.execute("_context_tool", arg1="hello"))
    self.assertEqual(result, "got hello")
    self.assertIs(received_ctx, mock_ctx)


class SchemaGenerationTest(absltest.TestCase):
  """Validates get_public_callable schema generation.

  Ensures that injectable parameters are hidden from the callable's
  signature so they don't appear in tool schemas sent to the model.
  """

  def test_public_callable_for_plain_tool(self):
    """Verifies that plain tools return themselves as the public callable.

    What: Checks that tools without ToolContext are returned as-is.
    Why: No schema modification should happen for regular tools.
    How: Calls get_public_callable and asserts identity.
    """
    runner = tool_runner.ToolRunner([_sample_tool])
    public = runner.get_public_callable("_sample_tool")
    self.assertIs(public, _sample_tool)

  def test_public_callable_hides_context_param(self):
    """Verifies that the ToolContext parameter is hidden from the signature.

    What: Checks that the public callable's signature lacks the context param.
    Why: The model must not see injectable parameters in the tool schema.
    How: Creates a tool with ToolContext, gets its public callable, and
    inspects the resulting signature.
    """
    import inspect  # pylint: disable=g-import-not-at-top
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    def _schema_tool(query: str, ctx: tool_context.ToolContext) -> str:
      return query

    runner = tool_runner.ToolRunner([_schema_tool])
    public = runner.get_public_callable("_schema_tool")
    sig = inspect.signature(public)
    self.assertIn("query", sig.parameters)
    self.assertNotIn("ctx", sig.parameters)

  def test_public_callable_for_unknown_tool_raises(self):
    """Verifies that get_public_callable raises for unknown tools.

    What: Checks error behavior for invalid tool names.
    Why: Consistent error handling with other ToolRunner methods.
    How: Calls get_public_callable with a non-existent name.
    """
    runner = tool_runner.ToolRunner()
    with self.assertRaises(KeyError):
      runner.get_public_callable("nonexistent")

  def test_public_callable_preserves_tool_with_schema_wrapper(self):
    import inspect  # pylint: disable=g-import-not-at-top
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top
    def _schema_tool(query: str, ctx: tool_context.ToolContext) -> str:
      del ctx
      return query

    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    tool = tool_runner.ToolWithSchema(_schema_tool, schema)
    runner = tool_runner.ToolRunner([tool])

    public = runner.get_public_callable("_schema_tool")
    self.assertIsInstance(public, tool_runner.ToolWithSchema)
    self.assertEqual(public.input_schema, schema)
    self.assertEqual(public.__name__, "_schema_tool")

    # Verify signature of the inner callable is modified
    sig = inspect.signature(public.fn)
    self.assertIn("query", sig.parameters)
    self.assertNotIn("ctx", sig.parameters)

  def test_public_callable_handles_nested_tool_with_schema_wrappers(self):
    import inspect  # pylint: disable=g-import-not-at-top
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top
    def _schema_tool(query: str, ctx: tool_context.ToolContext) -> str:
      del ctx
      return query

    inner_schema = {"type": "inner"}
    outer_schema = {"type": "outer"}
    tool = tool_runner.ToolWithSchema(
        tool_runner.ToolWithSchema(_schema_tool, inner_schema),
        outer_schema,
    )
    runner = tool_runner.ToolRunner([tool])

    public = runner.get_public_callable("_schema_tool")
    self.assertIsInstance(public, tool_runner.ToolWithSchema)
    self.assertEqual(public.input_schema, outer_schema)

    self.assertIsInstance(public.fn, tool_runner.ToolWithSchema)
    self.assertEqual(public.fn.input_schema, inner_schema)

    # Verify the innermost function has modified signature
    sig = inspect.signature(public.fn.fn)
    self.assertIn("query", sig.parameters)
    self.assertNotIn("ctx", sig.parameters)

  def test_public_callable_preserves_async_ness(self):
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    async def _schema_tool(query: str, ctx: tool_context.ToolContext) -> str:
      del ctx
      return query

    runner = tool_runner.ToolRunner([_schema_tool])
    public = runner.get_public_callable("_schema_tool")
    self.assertTrue(tool_runner._is_async(public))

  def test_public_callable_invokes_with_positional_and_keyword_args(self):
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    def _sync_tool(
        a: str, b: int, ctx: tool_context.ToolContext | None = None
    ) -> str:
      del ctx
      return f"{a}:{b}"

    async def _async_tool_with_ctx(
        a: str, b: int, ctx: tool_context.ToolContext | None = None
    ) -> str:
      del ctx
      return f"{a}:{b}"

    runner = tool_runner.ToolRunner([_sync_tool, _async_tool_with_ctx])

    sync_pub = runner.get_public_callable("_sync_tool")
    self.assertEqual(sync_pub("foo", 42), "foo:42")
    self.assertEqual(sync_pub(a="bar", b=10), "bar:10")

    async_pub = runner.get_public_callable("_async_tool_with_ctx")
    self.assertEqual(asyncio.run(async_pub("foo", 42)), "foo:42")
    self.assertEqual(asyncio.run(async_pub(a="bar", b=10)), "bar:10")

  def test_tool_with_schema_invokes_with_positional_and_keyword_args(self):
    def _my_tool(x: int, y: int) -> int:
      return x * y

    tool = tool_runner.ToolWithSchema(_my_tool, {"type": "object"})
    self.assertEqual(tool(3, 4), 12)
    self.assertEqual(tool(x=5, y=6), 30)

  def test_callable_object_context_injection(self):
    """Verifies that callable class instances receive ToolContext injection."""
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    class CallableWithContext:

      def __call__(
          self, query: str, ctx: tool_context.ToolContext
      ) -> str:
        ctx.set_state("called_with", query)
        return f"result:{query}"

    tool = CallableWithContext()
    runner = tool_runner.ToolRunner([tool])
    ctx = tool_context.ToolContext(mock.MagicMock())
    runner.set_context(ctx)

    result = asyncio.run(runner.execute("CallableWithContext", query="hello"))
    self.assertEqual(result, "result:hello")
    self.assertEqual(ctx.get_state("called_with"), "hello")

  def test_callable_object_argument_coercion(self):
    """Verifies that callable class instances have their arguments coerced based on type hints."""

    class CallableWithTypes:

      def __call__(self, count: int, enabled: bool) -> tuple[int, bool]:
        return count, enabled

    tool = CallableWithTypes()
    runner = tool_runner.ToolRunner([tool])

    result = asyncio.run(
        runner.execute("CallableWithTypes", count="42", enabled="true")
    )
    self.assertEqual(result, (42, True))

  def test_dataclass_callable_object_context_and_coercion(self):
    """Verifies dataclass tools with typed attributes resolve __call__ hints for context and coercion."""
    from google.antigravity.tools import tool_context  # pylint: disable=g-import-not-at-top

    @dataclasses.dataclass
    class StatefulDataclassTool:
      prefix: str
      limit: int

      def __call__(
          self, query: str, count: int, ctx: tool_context.ToolContext
      ) -> str:
        ctx.set_state("called_count", count)
        return f"{self.prefix}:{query}:{count}"

    tool = StatefulDataclassTool(prefix="STORE", limit=100)
    runner = tool_runner.ToolRunner([tool])
    ctx = tool_context.ToolContext(mock.MagicMock())
    runner.set_context(ctx)

    result = asyncio.run(
        runner.execute("StatefulDataclassTool", query="apples", count="25")
    )
    self.assertEqual(result, "STORE:apples:25")
    self.assertEqual(ctx.get_state("called_count"), 25)


if __name__ == "__main__":
  absltest.main()

