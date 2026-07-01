"""Shared graph state passed between agent nodes (LangGraph)."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    """State threaded through the planner -> retriever -> executor graph.

    `messages` accumulates across nodes; other keys are overwritten per step.
    """

    question: str
    plan: list[str]                       # planner output: ordered sub-steps
    retrieved: list[dict[str, Any]]       # retriever output: scored chunks
    tool_calls: list[dict[str, Any]]      # executor: tool invocations + results
    answer: str                           # final synthesized answer
    messages: Annotated[list[dict[str, Any]], add]   # running memory/log
