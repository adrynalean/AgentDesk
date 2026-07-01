"""Planner node: decompose the question into retrieval sub-queries."""
from __future__ import annotations

import re

from ..state import AgentState


def _subqueries(question: str) -> list[str]:
    """Split compound questions ('and', '?') into focused retrieval queries."""
    parts = re.split(r"\?|\band\b|;", question, flags=re.I)
    subs = [p.strip(" ,") for p in parts if len(p.strip()) > 3]
    return subs or [question.strip()]


def plan(state: AgentState) -> AgentState:
    subs = _subqueries(state["question"])
    return {
        **state,
        "plan": subs,
        "messages": [{"role": "planner", "content": subs}],
    }
