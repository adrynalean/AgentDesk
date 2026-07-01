"""Executor node: ReAct tool-calling + synthesize a grounded answer."""
from __future__ import annotations

from ..react import run_react
from ..state import AgentState


def execute(state: AgentState, llm) -> AgentState:
    result = run_react(llm, state["question"], state.get("retrieved", []))
    return {
        **state,
        "answer": result.answer,
        "tool_calls": [s.__dict__ for s in result.steps],
        "messages": [{"role": "executor", "content": "answer synthesized"}],
    }
