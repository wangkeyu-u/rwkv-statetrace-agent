"""Explicit allow-list of tools available to a task."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from ..models import ErrorCode, JSONValue, Observation
from .base import Tool, ToolContext, ToolError


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"Duplicate or empty tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def descriptions(self) -> list[dict[str, JSONValue]]:
        return [self._tools[name].describe() for name in sorted(self._tools)]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def execute(
        self, name: str, arguments: dict[str, Any], context: ToolContext
    ) -> Observation:
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return Observation(
                status="error",
                error_code=ErrorCode.UNKNOWN_TOOL,
                message=f"Unknown tool {name!r}. Use one of: {', '.join(self.names)}.",
                data={"allowed_tools": list(self.names)},
            )
        try:
            result = tool.run(arguments, context)
        except ToolError as exc:
            result = exc.as_observation()
        except Exception as exc:  # Boundary: do not crash the whole agent on a tool bug.
            result = Observation(
                status="error",
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Tool {name} failed safely: {type(exc).__name__}.",
            )
        result.duration_ms = (time.perf_counter() - started) * 1000
        return result


def default_registry() -> ToolRegistry:
    # Local imports avoid module cycles and keep custom registries lightweight.
    from .calculator import CalculatorTool
    from .files import ListFilesTool, ReadFileTool
    from .search import SearchCodeTool
    from .tests import RunTestsTool

    return ToolRegistry(
        [ListFilesTool(), SearchCodeTool(), ReadFileTool(), RunTestsTool(), CalculatorTool()]
    )
