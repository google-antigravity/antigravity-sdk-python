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

"""In-process tool runner for the Google Antigravity SDK.

Tools are Python callables that run directly in the SDK process. The
ToolRunner is a registry and executor — it holds references to tool
functions and invokes them by name when requested.

HOW tool calls reach the runner (callback server, direct invocation,
RPC bridge) is a connection strategy concern, not a tool runner concern.

ToolContext injection is handled here: at registration time the runner
inspects each tool's signature for a ``ToolContext``-typed parameter
and caches the result. At execution time, the context is injected
automatically. Schema generation (``get_public_callable``) strips
injectable parameters so the model never sees them.
"""

import asyncio
import functools
import inspect
import types as std_types
import typing
from typing import Any, Callable

import pydantic
from google.antigravity import types
from google.antigravity.tools import tool_context as tool_context_module


def _get_type_hints(target: Any) -> dict[str, Any]:
  """Resolves type hints for functions, methods, and callable objects."""
  if not inspect.isroutine(target) and hasattr(target, "__call__"):
    target = target.__call__
  try:
    return typing.get_type_hints(target)
  except (TypeError, NameError, AttributeError):
    return {}


def _find_context_param(fn: Callable[..., Any]) -> str | None:
  """Returns the name of the ToolContext-typed parameter, if any.

  Uses ``typing.get_type_hints`` to resolve annotations — including
  stringified ones from ``from __future__ import annotations`` — and
  checks for an exact match against ``ToolContext``.

  Args:
    fn: The callable to inspect. If it's a ``ToolWithSchema``, the inner ``.fn``
      is used.

  Returns:
    The parameter name, or None if no ToolContext parameter is found.
  """
  target = fn
  while isinstance(target, ToolWithSchema):
    target = target.fn
  hints = _get_type_hints(target)

  for name, ann in hints.items():
    if name == "return":
      continue
    if ann is tool_context_module.ToolContext:
      return name
    # Handle Optional[ToolContext] / ToolContext | None forms.
    origin = typing.get_origin(ann)
    if origin is typing.Union or origin is std_types.UnionType:
      if tool_context_module.ToolContext in typing.get_args(ann):
        return name

  # Fallback to direct inspection of signature parameter annotations
  try:
    sig = inspect.signature(target)
    for name, param in sig.parameters.items():
      ann = param.annotation
      if ann is inspect.Parameter.empty:
        continue
      if ann is tool_context_module.ToolContext or ann == "ToolContext":
        return name
      origin = typing.get_origin(ann)
      if origin is typing.Union or origin is std_types.UnionType:
        if tool_context_module.ToolContext in typing.get_args(ann):
          return name
  except (ValueError, TypeError):
    pass

  return None


def _make_public_callable(
    fn: Callable[..., Any], context_param: str
) -> Callable[..., Any]:
  """Returns a proxy callable with the injectable parameter removed.

  The proxy has the same ``__name__``, ``__doc__``, and a modified
  ``__signature__`` that excludes the ``context_param``. This is used
  for schema generation so the model never sees the injected parameter.

  Args:
    fn: The original callable.
    context_param: Name of the parameter to hide.

  Returns:
    A wrapper callable with a cleaned signature.
  """
  if isinstance(fn, ToolWithSchema):
    return ToolWithSchema(
        _make_public_callable(fn.fn, context_param), fn.input_schema
    )

  sig = inspect.signature(fn)
  new_params = [p for n, p in sig.parameters.items() if n != context_param]
  public_sig = sig.replace(parameters=new_params)

  if _is_async(fn):

    @functools.wraps(fn)
    async def _proxy(*args: Any, **kwargs: Any) -> Any:
      return await fn(*args, **kwargs)

  else:

    @functools.wraps(fn)
    def _proxy(*args: Any, **kwargs: Any) -> Any:
      return fn(*args, **kwargs)

  setattr(_proxy, "__signature__", public_sig)
  return _proxy


class ToolWithSchema:
  """Wrapper for callables with an explicit JSON Schema."""

  def __init__(self, fn: Callable[..., Any], input_schema: dict[str, Any]):
    self.fn = fn
    self.input_schema = input_schema
    self.__name__ = getattr(fn, "__name__", None) or type(fn).__name__
    self.__doc__ = getattr(fn, "__doc__", None)

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self.fn(*args, **kwargs)


def _is_async(callable_obj: Any) -> bool:
  """Returns True if the callable is async (coroutine function or __call__)."""
  if isinstance(callable_obj, ToolWithSchema):
    return _is_async(callable_obj.fn)
  return inspect.iscoroutinefunction(callable_obj) or (
      hasattr(callable_obj, "__call__")
      and inspect.iscoroutinefunction(callable_obj.__call__)
  )


class ToolRunner:
  """Registry and executor for in-process Python tools.

  Tools are registered by name and executed on demand. Both sync and async
  tools are supported. Tools that declare a ``ToolContext``-typed parameter
  receive the context automatically at execution time.
  """

  def __init__(self, tools: list[types.PythonTool] | None = None):
    self._tools: dict[str, types.PythonTool] = {}
    self._context: tool_context_module.ToolContext | None = None
    # Maps tool name → parameter name for ToolContext injection.
    # Populated at registration time to avoid per-call introspection.
    self._context_params: dict[str, str] = {}
    if tools:
      for tool in tools:
        self.register(tool)

  def set_context(self, ctx: tool_context_module.ToolContext) -> None:
    """Sets the ToolContext for injection into tools that request it.

    Args:
      ctx: The ToolContext to inject.
    """
    self._context = ctx

  def register(self, tool: types.PythonTool, name: str | None = None) -> None:
    """Registers a tool by name.

    At registration time, the tool's signature is inspected for a
    ``ToolContext``-typed parameter. If found, the parameter name is
    cached for injection at execution time.

    Args:
      tool: The callable to register.
      name: Optional name override. Defaults to tool.__name__.

    Raises:
      ValueError: If a tool with the same name is already registered.
    """
    tool_name = name or getattr(tool, "__name__", None) or type(tool).__name__
    if tool_name in self._tools:
      raise ValueError(f"Tool '{tool_name}' is already registered.")
    self._tools[tool_name] = tool

    # Cache the ToolContext parameter name (if any) at registration time.
    ctx_param = _find_context_param(tool)
    if ctx_param is not None:
      self._context_params[tool_name] = ctx_param

  def unregister(self, name: str) -> None:
    """Removes a tool by name.

    Args:
      name: The name of the tool to remove.

    Raises:
      KeyError: If no tool with the given name is registered.
    """
    if name not in self._tools:
      raise KeyError(f"Tool '{name}' is not registered.")
    del self._tools[name]
    self._context_params.pop(name, None)

  @property
  def tool_names(self) -> list[str]:
    """The names of all registered tools."""
    return list(self._tools.keys())

  @property
  def tools(self) -> dict[str, types.PythonTool]:
    """A copy of the registered tools dictionary."""
    return dict(self._tools)

  def get_public_callable(self, tool_name: str) -> Callable[..., Any]:
    """Returns a callable with injectable params hidden from the signature.

    Connections use this to generate tool schemas for the model without
    exposing internal-only parameters like ``ToolContext``.

    Args:
      tool_name: The name of the registered tool.

    Returns:
      The original callable if no context parameter, otherwise a proxy
      callable with the context parameter removed from the signature.

    Raises:
      KeyError: If no tool with the given name is registered.
    """
    if tool_name not in self._tools:
      raise KeyError(f"Tool '{tool_name}' is not registered.")
    tool = self._tools[tool_name]
    ctx_param = self._context_params.get(tool_name)
    if ctx_param is None:
      return tool
    return _make_public_callable(tool, ctx_param)

  async def _execute_fn(self, fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Executes a callable, running sync functions in a separate thread."""
    if not _is_async(fn):
      result = await asyncio.to_thread(fn, **kwargs)
    else:
      result = fn(**kwargs)

    if asyncio.iscoroutine(result):
      return await result
    return result

  def _inject_context(
      self, tool_name: str, kwargs: dict[str, Any]
  ) -> dict[str, Any]:
    """Returns kwargs augmented with ToolContext if the tool requests it.

    Returns a new dict when injection occurs; returns the original
    dict unchanged otherwise (no unnecessary copies).

    Args:
      tool_name: The registered tool name.
      kwargs: The original keyword arguments.

    Returns:
      The kwargs dict, potentially augmented with the ToolContext.
    """
    ctx_param = self._context_params.get(tool_name)
    if ctx_param is not None and self._context is not None:
      if ctx_param not in kwargs:
        return {**kwargs, ctx_param: self._context}
    return kwargs

  def _coerce_args(
      self, fn: Callable[..., Any], kwargs: dict[str, Any]
  ) -> dict[str, Any]:
    """Coerces keyword arguments to match function type annotations using pydantic."""
    target = fn
    while isinstance(target, ToolWithSchema):
      target = target.fn
    try:
      sig = inspect.signature(target)
    except (ValueError, TypeError):
      return kwargs

    hints = _get_type_hints(target)

    coerced = {}
    for name, param in sig.parameters.items():
      if name not in kwargs:
        continue
      if param.kind in (
          inspect.Parameter.VAR_POSITIONAL,
          inspect.Parameter.VAR_KEYWORD,
      ):
        coerced[name] = kwargs[name]
        continue
      val = kwargs[name]
      ann = hints.get(name, param.annotation)
      if ann is inspect.Parameter.empty or val is None:
        coerced[name] = val
        continue

      try:
        adapter = pydantic.TypeAdapter(ann)
        coerced[name] = adapter.validate_python(val)
      except Exception:  # pylint: disable=broad-except
        coerced[name] = val

    # Retain any extra arguments passed that were not in signature (e.g. kwargs)
    for k, v in kwargs.items():
      if k not in coerced:
        coerced[k] = v

    return coerced

  async def execute(self, tool_name: str, **kwargs: Any) -> Any:
    """Executes a registered tool by name.

    If the tool declares a ``ToolContext`` parameter and a context has
    been set via ``set_context()``, the context is injected automatically.

    Args:
      tool_name: The name of the tool to execute.
      **kwargs: Arguments to pass to the tool.

    Returns:
      The tool's return value.

    Raises:
      KeyError: If no tool with the given name is registered.
    """
    if tool_name not in self._tools:
      raise KeyError(f"Tool '{tool_name}' is not registered.")

    tool_fn = self._tools[tool_name]
    kwargs = self._coerce_args(tool_fn, kwargs)
    kwargs = self._inject_context(tool_name, kwargs)
    return await self._execute_fn(tool_fn, **kwargs)

  async def process_tool_calls(
      self,
      tool_calls: list[types.ToolCall],
  ) -> list[types.ToolResult]:
    """Executes a batch of tool calls concurrently and returns structured results.

    Tool calls are executed in parallel via ``asyncio.gather``.  Unknown
    tools and execution failures produce ToolResult with an error message
    rather than raising.

    Note: tools execute concurrently; callers must not depend on
    sequential side-effect ordering.

    Args:
      tool_calls: List of ToolCall objects.

    Returns:
      A list of ToolResult, one per input tool call, in the same order.
    """

    async def _execute_one(tc: types.ToolCall) -> types.ToolResult:
      # The entire body is wrapped in try/except so that nothing can
      # escape and cause asyncio.gather to cancel sibling tasks.
      try:
        if tc.name not in self._tools:
          return types.ToolResult(
              id=tc.id,
              step_id=tc.step_id,
              server_name=tc.server_name,
              name=tc.name,
              error=f"Unknown tool: '{tc.name}'",
          )
        result = await self.execute(tc.name, **tc.args)
        return types.ToolResult(
            id=tc.id,
            step_id=tc.step_id,
            server_name=tc.server_name,
            name=tc.name,
            result=result,
        )
      except Exception as e:  # pylint: disable=broad-except
        return types.ToolResult(
            id=tc.id,
            step_id=tc.step_id,
            server_name=tc.server_name,
            name=tc.name,
            error=str(e),
            exception=e,
        )

    return list(await asyncio.gather(*[_execute_one(tc) for tc in tool_calls]))
