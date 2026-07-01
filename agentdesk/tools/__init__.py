"""Tool registry for the executor's tool-calling loop.

Each tool is a callable with a name, description, and JSON-serializable I/O so it can be
exposed to the model as a callable function.
"""
from __future__ import annotations

from .calculator import calculator
from .web_search import web_search

TOOLS = {
    "calculator": calculator,
    "web_search": web_search,
}

__all__ = ["TOOLS", "calculator", "web_search"]
