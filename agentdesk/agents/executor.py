"""Executor node: run tools (tool-calling) and synthesize the grounded answer."""
from __future__ import annotations

from ..state import AgentState
from ..tools import TOOLS


def execute(state: AgentState) -> AgentState:
    """Run any needed tools, then synthesize an answer from retrieved context.

    TODO(M3): implement the tool-calling loop — let the model choose tools from
    TOOLS, execute them, feed results back until it emits a final answer.
    """
    context = state.get("retrieved", [])
    _ = TOOLS  # available tools for the model to call
    answer = (
        "TODO(M3): synthesize grounded answer from "
        f"{len(context)} retrieved chunks + tool results."
    )
    return {
        **state,
        "tool_calls": [],
        "answer": answer,
        "messages": [{"role": "executor", "content": "answer drafted"}],
    }
