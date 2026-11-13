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

"""Helper for converting Python callables into function tool schema dicts."""

import functools
import inspect
from typing import Any, Callable

from google.genai import types as genai_types

from google.antigravity.tools import schema_utils
from google.antigravity.tools import tool_runner as t_runner


# External callers still depend on importing callable_to_function_tool_dict from
# this module.
# TODO: Migrate external callers and clean up this module.
def callable_to_function_tool_dict(
    fn: Callable[..., Any],
    tool_runner: t_runner.ToolRunner | None = None,
) -> dict[str, Any]:
  """Converts a Python callable to a function tool schema dict."""
  if isinstance(fn, t_runner.ToolWithSchema):
    fn_dict: dict[str, Any] = {
        "type": "function",
        "name": getattr(fn, "__name__", ""),
    }
    if fn.__doc__:
      fn_dict["description"] = fn.__doc__
    fn_dict["parameters"] = schema_utils.normalize_schema(fn.input_schema)
    return fn_dict

  target_fn: Callable[..., Any] = fn
  tool_name = t_runner.get_tool_name(fn)
  # When a tool uses ToolContext dependency injection, ToolRunner strips the
  # injected parameter via get_public_callable() so the model's JSON schema only
  # exposes user-facing arguments. tool_runner=None is an optional fallback for
  # standalone callables without injected parameters.
  if tool_runner is not None and tool_name in tool_runner.tools:
    target_fn = tool_runner.get_public_callable(tool_name)

  if not hasattr(target_fn, "__name__"):
    orig_fn = target_fn
    base_fn: Any = orig_fn
    while isinstance(base_fn, functools.partial):
      base_fn = base_fn.func
    routine_fn = (
        base_fn if inspect.isroutine(base_fn) else getattr(base_fn, "__call__")
    )

    @functools.wraps(routine_fn)
    def wrapped() -> None:
      """Introspection stub preserving __wrapped__ for deferred type hints."""

    wrapped.__name__ = tool_name
    wrapped.__doc__ = t_runner.get_tool_doc(fn) or None
    try:
      setattr(wrapped, "__signature__", inspect.signature(orig_fn))
    except (ValueError, TypeError):
      setattr(wrapped, "__signature__", inspect.Signature())
    target_fn = wrapped

  decl = genai_types.FunctionDeclaration.from_callable_with_api_option(
      callable=target_fn,
      api_option="GEMINI_API",
  )
  if decl.parameters:
    parameters = decl.parameters.model_dump(exclude_none=True)
  elif decl.parameters_json_schema:
    parameters = decl.parameters_json_schema
  else:
    parameters = {"type": "object", "properties": {}}

  fn_dict = {
      "type": "function",
      "name": decl.name or "",
  }
  if decl.description:
    fn_dict["description"] = decl.description
  fn_dict["parameters"] = schema_utils.normalize_schema(parameters)
  return fn_dict
