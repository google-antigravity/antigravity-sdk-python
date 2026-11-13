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

"""Unit tests for interactions_config_converter."""

import dataclasses
import functools
from typing import Any

from absl.testing import absltest

from google.antigravity import types
from google.antigravity.connections.local import interactions_config_converter
from google.antigravity.tools import tool_context
from google.antigravity.tools import tool_runner

_DeferredSection = types.SystemInstructionSection


def _deferred_tag_message(prefix: "str", body: "str") -> "str":
  """Tags a message with a fixed prefix."""
  return f"{prefix}:{body}"


class _DeferredFunctor:

  def __call__(self, value: "int", section: "_DeferredSection") -> "str":
    """Formats a section with an integer value."""
    return f"{value}:{section.content}"


class _ClassDocFunctor:
  """Class-level functor documentation."""

  def __call__(self, query: "str") -> "str":
    return query


@dataclasses.dataclass
class _DataclassNoDocFunctor:
  scale: int

  def __call__(self, count: "int") -> "int":
    return count * self.scale


class _UninspectableCallable:

  @property
  def __signature__(self):
    raise ValueError("no signature")

  def __call__(self, *args: Any, **kwargs: Any) -> None:
    del args, kwargs


class InteractionsConfigConverterTest(absltest.TestCase):

  def test_callable_to_function_tool_dict_variants(self):
    def add_numbers(a: int, b: int) -> int:
      """Adds two numbers."""
      return a + b

    def ctx_tool(ctx: tool_context.ToolContext, query: str) -> str:
      """Tool with injected ToolContext."""
      del ctx
      return query

    def schema_fn(y: str) -> str:
      """Tool with explicit schema."""
      return y

    class _CallableObj:
      """Callable object without __name__."""

      def __call__(self, x: int) -> int:
        return x + 1

    fn_dict = interactions_config_converter.callable_to_function_tool_dict(
        add_numbers
    )
    self.assertEqual(fn_dict["type"], "function")
    self.assertEqual(fn_dict["name"], "add_numbers")
    self.assertEqual(fn_dict["description"], "Adds two numbers.")
    self.assertIn("a", fn_dict["parameters"]["properties"])
    self.assertIn("b", fn_dict["parameters"]["properties"])

    runner = tool_runner.ToolRunner([ctx_tool])
    ctx_dict = interactions_config_converter.callable_to_function_tool_dict(
        ctx_tool, tool_runner=runner
    )
    self.assertEqual(ctx_dict["name"], "ctx_tool")
    self.assertEqual(ctx_dict["description"], "Tool with injected ToolContext.")
    self.assertEqual(list(ctx_dict["parameters"]["properties"]), ["query"])

    callable_obj = _CallableObj()
    obj_dict = interactions_config_converter.callable_to_function_tool_dict(
        callable_obj
    )
    self.assertEqual(obj_dict["name"], "_CallableObj")
    self.assertIn("x", obj_dict["parameters"]["properties"])

    schema_tool = tool_runner.ToolWithSchema(
        schema_fn,
        input_schema={
            "type": "OBJECT",
            "properties": {"y": {"type": "STRING"}},
        },
    )
    schema_dict = interactions_config_converter.callable_to_function_tool_dict(
        schema_tool
    )
    self.assertEqual(schema_dict["name"], "schema_fn")
    self.assertEqual(schema_dict["description"], "Tool with explicit schema.")
    self.assertEqual(
        schema_dict["parameters"],
        {"type": "object", "properties": {"y": {"type": "string"}}},
    )

  def test_callable_functor_and_partial_preserve_deferred_annotations_and_docs(
      self,
  ):
    """Verifies schema extraction for functors and partials with deferred hints.

    What: Converts callable class instances, functools.partial wrappers, and
      uninspectable callables whose annotations are PEP 563 strings.
    Why: Wrapping non-function callables without functools.wraps drops the
      underlying __call__ or target function globals/annotations, causing
      FunctionDeclaration schema parsing to raise KeyError on stringified hints
      and lose __call__ or partial docstrings/names.
    How: Asserts resolved parameter schemas, tool names, and descriptions across
      functor, dataclass functor, functools.partial, and signature-less cases.
    """
    functor = _DeferredFunctor()
    tr = tool_runner.ToolRunner([functor])
    for runner in (None, tr):
      fn_dict = interactions_config_converter.callable_to_function_tool_dict(
          functor, tool_runner=runner
      )
      self.assertEqual(fn_dict["name"], "_DeferredFunctor")
      self.assertEqual(
          fn_dict["description"], "Formats a section with an integer value."
      )
      params = fn_dict["parameters"]
      self.assertEqual(params["type"], "object")
      self.assertEqual(params["properties"]["value"]["type"], "integer")
      self.assertEqual(params["properties"]["section"]["type"], "object")
      self.assertIn("content", params["properties"]["section"]["properties"])
      self.assertEqual(params["required"], ["value", "section"])

    class_doc_dict = (
        interactions_config_converter.callable_to_function_tool_dict(
            _ClassDocFunctor()
        )
    )
    self.assertEqual(class_doc_dict["name"], "_ClassDocFunctor")
    self.assertEqual(
        class_doc_dict["description"], "Class-level functor documentation."
    )
    self.assertEqual(
        class_doc_dict["parameters"]["properties"]["query"]["type"], "string"
    )

    dataclass_dict = (
        interactions_config_converter.callable_to_function_tool_dict(
            _DataclassNoDocFunctor(scale=3)
        )
    )
    self.assertEqual(dataclass_dict["name"], "_DataclassNoDocFunctor")
    self.assertNotIn("description", dataclass_dict)
    self.assertEqual(
        dataclass_dict["parameters"]["properties"]["count"]["type"], "integer"
    )

    partial_fn = functools.partial(_deferred_tag_message, "INFO")
    partial_dict = interactions_config_converter.callable_to_function_tool_dict(
        partial_fn
    )
    self.assertEqual(partial_dict["name"], "_deferred_tag_message")
    self.assertEqual(
        partial_dict["description"], "Tags a message with a fixed prefix."
    )
    self.assertEqual(set(partial_dict["parameters"]["properties"]), {"body"})
    self.assertEqual(
        partial_dict["parameters"]["properties"]["body"]["type"], "string"
    )

    uninspectable_dict = (
        interactions_config_converter.callable_to_function_tool_dict(
            _UninspectableCallable()
        )
    )
    self.assertEqual(uninspectable_dict["name"], "_UninspectableCallable")
    self.assertEqual(
        uninspectable_dict["parameters"],
        {"type": "object", "properties": {}},
    )


if __name__ == "__main__":
  absltest.main()
