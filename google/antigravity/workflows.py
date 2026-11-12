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

"""Authoring utilities and typed stubs for `run_workflow` scripts."""

from __future__ import annotations

import ast
from collections.abc import Awaitable, Callable, Coroutine, Generator, Iterable
import dataclasses
import inspect
import sys
from typing import Any, Literal

from google.antigravity import beta as _beta_lib

# VCS workspace isolation mode for a spawned workflow agent:
# - "inherit": Run in the parent agent's existing workspace directory.
# - "branch": Fork an isolated VCS workspace copy (e.g., git worktree) that is
#   cleaned up when the workflow finishes.
# - "share": Create a workspace sharing the parent's underlying repository
#   storage (e.g., git worktree / hg share) that is cleaned up when the
#   workflow finishes.
WorkspaceMode = Literal["inherit", "branch", "share"]

_PRIMITIVE_NAMES: frozenset[str] = frozenset({
    "phase",
    "log",
    "agent",
    "parallel",
    "pipeline",
    "call_tool",
    "ToolError",
    "args",
})

_WORKFLOW_MODULE_NAMES: frozenset[str] = frozenset({
    "workflows",
    "workflow",
})


class WorkflowError(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when a workflow script fails validation or runtime execution."""


class ToolError(WorkflowError):
  """Raised inside a workflow script when a `call_tool` invocation fails or is denied.

  Attributes:
    tool: Name of the tool that failed or was denied.
    message: Error message returned by the harness.
  """

  def __init__(self, tool: str, message: str) -> None:
    super().__init__(f"{tool}: {message}")
    self.tool = tool
    self.message = message


# ---------------------------------------------------------------------------
# Compile-time type stubs for `run_workflow` primitives.
#
# These functions exist so that `@workflows.define` functions can be written in
# normal Python modules with full IDE autocomplete and static type-checking
# (`workflows.phase(...)`, `await workflows.parallel(...)`, etc.). At decoration
# time, `extract_workflow_source()` extracts the function body's AST and
# rewrites `workflows.<primitive>(...)` into Antigravity workflow code.
#
# Calling these stubs directly in the host SDK Python process is a user error
# and raises RuntimeError.
# ---------------------------------------------------------------------------

# Compile-time stub for the `args` dictionary injected into workflow scripts
# when `Args` is passed to `run_workflow`.
args: dict[str, Any] = {}


def _outside_runtime_error(name: str) -> RuntimeError:
  return RuntimeError(
      f"workflows.{name}() is a compile-time stub for the `run_workflow` tool"
      " and cannot be called directly in the host Python process. Call it"
      " inside a `@workflows.define` function and execute the workflow via"
      " `await agent.beta.run_workflow(workflow=...)`."
  )


class WorkflowAgent:
  """Compile-time stub for a single-use or multi-turn workflow agent.

  Inside a `@workflows.define` function:
  - Single-use: `res = await workflows.agent("Prompt", schema={...})`
  - Multi-turn:
    ```python
    reviewer = workflows.agent("Standing instructions", role="Reviewer")
    turn1 = await reviewer("First turn", schema={"type": "string"})
    turn2 = await reviewer("Second turn", schema={"type": "string"})
    ```
  """

  def __await__(self) -> Generator[Any, None, Any]:
    raise _outside_runtime_error("agent")

  def __call__(
      self,
      prompt: str,
      schema: Any = None,
  ) -> Coroutine[Any, Any, Any]:
    del prompt, schema
    raise _outside_runtime_error("agent")


def log(*values: Any) -> None:
  """Compile-time stub: emits a progress log line to the current workflow phase."""
  if not values:
    raise TypeError("log: requires at least 1 argument")
  raise _outside_runtime_error("log")


def phase(name: str) -> str:
  """Compile-time stub: starts a new named phase in the workflow execution."""
  del name
  raise _outside_runtime_error("phase")


def agent(
    prompt: str | None = None,
    schema: Any = None,
    role: str | None = None,
    type_name: str = "self",
    workspace: WorkspaceMode = "inherit",
) -> WorkflowAgent:
  """Compile-time stub: defines a single-use or multi-turn workflow agent.

  Args:
    prompt: Task prompt (for a single-turn `await workflows.agent(...)` call) or
      standing instructions (when binding a multi-turn `WorkflowAgent` variable
      before calling `await handle(...)`).
    schema: Optional JSON Schema dict constraining the agent's structured output
      (only used for single-turn `await workflows.agent(...)` calls; for
      multi-turn agents, pass `schema` on each `await handle(prompt,
      schema=...)` turn).
    role: Optional human-readable display label for this specific agent
      invocation in workflow progress logs (e.g., `"Audit auth.py"` when running
      multiple parallel calls that share the same `type_name`). Defaults to a
      truncated prefix of `prompt`.
    type_name: Name of the agent definition to spawn. Defaults to `"self"`
      (cloning the parent agent's configuration), or matches the `name` of a
      `types.SubagentConfig` registered in `LocalAgentConfig(subagents=[...])`.
      Use `types.SubagentConfig` to configure an explicit `model` name (e.g.,
      `model="gemini-4-hydrogen"`), `system_instructions`, `capabilities`,
      `tools`, or `skills_config` for the workflow agent.
    workspace: VCS workspace isolation mode for the spawned agent: `"inherit"`
      (run in the parent workspace), `"branch"` (fork an isolated workspace copy
      cleaned up when the workflow finishes), or `"share"` (share the underlying
      repository directory in a separate working tree).

  Returns:
    A `WorkflowAgent` handle that can be awaited directly for a single-turn
    execution or called across multiple turns (`await handle(prompt,
    schema=...)`).

  Raises:
    RuntimeError: Always, if called directly in the host Python process rather
      than inside a `@workflows.define` function.
  """
  del prompt, schema, role, type_name, workspace
  raise _outside_runtime_error("agent")


def call_tool(name: str, **tool_args: Any) -> Coroutine[Any, Any, str]:
  """Compile-time stub: invokes an agent tool programmatically and returns its output."""
  del tool_args
  if not isinstance(name, str) or not name.strip():
    raise ValueError("call_tool: name must be a non-empty string")
  raise _outside_runtime_error("call_tool")


def parallel(
    items: Iterable[Any],
    fn: Callable[[Any], Awaitable[Any]],
) -> Coroutine[Any, Any, list[Any]]:
  """Compile-time stub: runs `await fn(item)` for each item concurrently."""
  del items, fn
  raise _outside_runtime_error("parallel")


def pipeline(
    items: Iterable[Any],
    fn: Callable[[Any], Awaitable[Any]],
) -> Coroutine[Any, Any, list[Any]]:
  """Compile-time stub: runs `await fn(item)` for each item one by one."""
  del items, fn
  raise _outside_runtime_error("pipeline")


def _is_workflow_module_ref(node: ast.AST) -> bool:
  """Returns True if `node` represents `workflows`, `workflow`, or `beta.workflows`."""
  if (
      isinstance(node, ast.Name)
      and node.id in _WORKFLOW_MODULE_NAMES
      and isinstance(node.ctx, ast.Load)
  ):
    return True
  if (
      isinstance(node, ast.Attribute)
      and isinstance(node.value, ast.Name)
      and node.value.id == "beta"
      and isinstance(node.value.ctx, ast.Load)
      and node.attr in _WORKFLOW_MODULE_NAMES
      and isinstance(node.ctx, ast.Load)
  ):
    return True
  return False


class _WorkflowQualifierStripper(ast.NodeTransformer):
  """Rewrites `(beta.)workflows.<primitive>(...)` AST nodes into Antigravity workflow code.

  When a user authors a function with `@workflows.define` or
  `@beta.workflows.define`, they write `await workflows.parallel(...)` or
  `await beta.workflows.parallel(...)` so static typecheckers resolve the
  symbol; this transformer strips the module attribute prefix when generating
  the standalone workflow script.
  """

  def __init__(self) -> None:
    super().__init__()
    self.modified = False

  def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # pylint: disable=invalid-name
    """Rewrites `(beta.)workflows.<primitive>` loads to bare `<primitive>` names."""
    self.generic_visit(node)
    if (
        _is_workflow_module_ref(node.value)
        and node.attr in _PRIMITIVE_NAMES
        and isinstance(node.ctx, ast.Load)
    ):
      self.modified = True
      return ast.copy_location(
          ast.Name(id=node.attr, ctx=ast.Load()),
          node,
      )
    return node


def _is_print_call(node: ast.AST) -> bool:
  """Returns whether `node` is a call to the built-in `print` function."""
  if not isinstance(node, ast.Call):
    return False
  func = node.func
  if isinstance(func, ast.Name) and func.id == "print":
    return True
  if (
      isinstance(func, ast.Attribute)
      and func.attr == "print"
      and isinstance(func.value, ast.Name)
      and func.value.id in ("builtins", "__builtins__")
  ):
    return True
  return False


def _check_no_top_level_return_or_yield(nodes: list[ast.stmt]) -> None:
  """Rejects `return` at the top level and `yield` or `print()` anywhere."""
  stack: list[ast.AST] = list(nodes)
  while stack:
    node = stack.pop()
    if _is_print_call(node):
      raise WorkflowError(
          "print() is not allowed in workflow scripts; use"
          " workflows.log(...) to emit output"
      )
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      # `return` is valid inside nested helper functions (such as `async def`
      # stage helpers passed to `pipeline` or `parallel`).
      for sub in ast.walk(node):
        if _is_print_call(sub):
          raise WorkflowError(
              "print() is not allowed in workflow scripts; use"
              " workflows.log(...) to emit output"
          )
        if isinstance(sub, (ast.Yield, ast.YieldFrom)):
          raise WorkflowError(
              "return/yield statements are not allowed in workflow scripts; use"
              " workflows.log(...) to emit output"
          )
      continue
    if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
      raise WorkflowError(
          "return/yield statements are not allowed in workflow scripts; use"
          " workflows.log(...) to emit output"
      )
    stack.extend(ast.iter_child_nodes(node))


def validate_workflow_source(source: str) -> None:
  """Validates Python workflow script source against `run_workflow` syntax rules.

  These checks validate the script at `@workflows.define` decoration time so
  disallowed constructs fail fast before execution:
  - Top-level `ast.Return` and `ast.Yield` / `ast.YieldFrom`: disallowed at the
    top level of a workflow script (`return` inside nested `def` / `async def`
    stage helpers is permitted).
  - Calls to `print(...)`: disallowed in workflow scripts (`workflows.log(...)`
    should be used instead).
  - Top-level `await` compilation (`ast.PyCF_ALLOW_TOP_LEVEL_AWAIT`).

  Args:
    source: Python source code of the workflow script.

  Raises:
    WorkflowError: If the source has a syntax error or uses top-level
      `return`/`yield` or `print()`.
  """
  try:
    tree = ast.parse(source, "<workflow>")
    _check_no_top_level_return_or_yield(tree.body)
    compile(tree, "<workflow>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
  except (SyntaxError, ValueError, TypeError) as e:
    raise WorkflowError(f"workflow syntax error: {e}") from e


def extract_workflow_source(
    fn_or_def: Callable[[], Any] | WorkflowDefinition,
) -> str:
  """Extracts and validates the standalone script source from a workflow function.

  Given a `@workflows.define` function such as:
    ```python
    @workflows.define
    async def my_workflow():
      workflows.phase("Step 1")
      res = await workflows.agent("Check status", schema={"type": "string"})
      workflows.log(res)
    ```
  this function extracts the function body, strips the docstring, and rewrites
  `(beta.)workflows.<primitive>(...)` calls into Antigravity workflow code:
    ```python
    phase('Step 1')
    res = await agent('Check status', schema={'type': 'string'})
    log(res)
    ```

  Args:
    fn_or_def: A `@workflows.define` definition or a zero-argument Python
      function (`async def` or `def`) whose body forms the workflow script.

  Returns:
    The validated Python source string ready to be written to a workflow script.

  Raises:
    WorkflowError: If the function takes arguments, its source cannot be
      extracted, or its body violates `run_workflow` AST restrictions.
  """
  if isinstance(fn_or_def, WorkflowDefinition):
    return fn_or_def.source

  if not callable(fn_or_def):
    raise TypeError(
        "Expected a callable or WorkflowDefinition, got"
        f" {type(fn_or_def).__name__}"
    )

  sig = inspect.signature(fn_or_def)
  if sig.parameters:
    raise WorkflowError(
        "workflow functions cannot require positional/keyword arguments; got"
        f" parameters {list(sig.parameters)}"
    )

  try:
    raw_source = inspect.getsource(fn_or_def)
  except Exception as e:
    raise WorkflowError(
        f"cannot inspect source of workflow function {fn_or_def!r}: {e}"
    ) from e

  first_line = next(
      (line for line in raw_source.splitlines() if line.strip()), ""
  )
  is_indented = first_line[:1].isspace()
  source_to_parse = f"if True:\n{raw_source}" if is_indented else raw_source
  try:
    mod_tree = ast.parse(source_to_parse)
  except (SyntaxError, ValueError, TypeError) as e:
    raise WorkflowError(f"workflow syntax error: {e}") from e

  stmts = (
      mod_tree.body[0].body
      if is_indented and mod_tree.body and isinstance(mod_tree.body[0], ast.If)
      else mod_tree.body
  )
  if not stmts or not isinstance(
      stmts[0], (ast.FunctionDef, ast.AsyncFunctionDef)
  ):
    raise WorkflowError("expected a function definition for workflow")

  fn_node = stmts[0]

  body_nodes = list(fn_node.body)
  # Strip leading docstring expression if present.
  if (
      body_nodes
      and isinstance(body_nodes[0], ast.Expr)
      and isinstance(body_nodes[0].value, ast.Constant)
      and isinstance(body_nodes[0].value.value, str)
  ):
    body_nodes = body_nodes[1:]

  if not body_nodes:
    raise WorkflowError("workflow function body cannot be empty")

  body_module = ast.Module(body=body_nodes, type_ignores=[])
  stripper = _WorkflowQualifierStripper()
  transformed_module = stripper.visit(body_module)
  ast.fix_missing_locations(transformed_module)
  script_source = ast.unparse(transformed_module) + "\n"

  validate_workflow_source(script_source)
  return script_source


@dataclasses.dataclass(frozen=True)
class WorkflowDefinition:
  """A validated workflow script extracted from a `@workflows.define` function.

  Attributes:
    name: Function name of the workflow.
    description: Docstring of the workflow function, if any.
    source: Validated standalone Python source for the `run_workflow` tool.
    fn: Original Python function.
  """

  name: str
  description: str
  source: str
  fn: Callable[[], Any]

  def __call__(self, *call_args: Any, **kwargs: Any) -> Any:
    return self.fn(*call_args, **kwargs)


def define(fn: Callable[[], Any]) -> WorkflowDefinition:
  r"""Decorator that validates a Python function as a `run_workflow` script.

  Example:
    ```python
    from google.antigravity.beta import workflows

    @workflows.define
    async def audit_repo():
      \"\"\"Audit repository modules in parallel.\"\"\"
      workflows.phase("Discover")
      modules = await workflows.agent(
          "List top-level modules",
          schema={"type": "array", "items": {"type": "string"}},
      )
      workflows.phase("Audit")
      findings = await workflows.parallel(
          modules,
          lambda m: workflows.agent(
              f"Audit module {m}",
              schema={"type": "string"},
              role=f"Audit {m}",
          ),
      )
      for f in findings:
        workflows.log(f)
    ```

  Args:
    fn: Zero-argument Python function (`async def` or `def`) using
      `run_workflow` primitives (`phase`, `log`, `agent`, `parallel`,
      `pipeline`).

  Returns:
    A `WorkflowDefinition` containing the validated script source.
  """
  source = extract_workflow_source(fn)
  doc = (inspect.getdoc(fn) or "").strip()
  return WorkflowDefinition(
      name=getattr(fn, "__name__", "workflow"),
      description=doc,
      source=source,
      fn=fn,
  )


_beta_lib.beta(sys.modules[__name__])
