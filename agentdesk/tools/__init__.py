"""Tool registry for the executor's tool-calling loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .calculator import calculator
from .web_search import web_search


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], str]


TOOLS: dict[str, Tool] = {
    "calculator": Tool("calculator", "Evaluate a basic arithmetic expression.", calculator),
    "web_search": Tool("web_search", "Search the web for a short factual answer.", web_search),
}


def tool_descriptions() -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in TOOLS.values())


__all__ = ["Tool", "TOOLS", "tool_descriptions", "calculator", "web_search"]
