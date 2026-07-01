"""Planner node: decompose the question into ordered sub-steps."""
from __future__ import annotations

from ..state import AgentState

PLANNER_SYSTEM = (
    "You are a planning agent. Break the user's question into a short ordered list "
    "of retrieval/tool steps needed to answer it faithfully. Return concise steps."
)


def plan(state: AgentState) -> AgentState:
    """Produce state['plan'].

    TODO(M2): call the chat model with PLANNER_SYSTEM + question, parse steps.
    """
    question = state["question"]
    plan_steps = [f"retrieve context for: {question}", "synthesize grounded answer"]
    return {
        **state,
        "plan": plan_steps,
        "messages": [{"role": "planner", "content": plan_steps}],
    }
