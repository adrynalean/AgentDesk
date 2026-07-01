"""LangGraph wiring: planner -> retriever -> executor, with shared memory.

Uses langgraph.StateGraph when installed; otherwise runs the same node functions
in sequence so the agent works in any environment.
"""
from __future__ import annotations

from .agents.executor import execute
from .agents.planner import plan
from .agents.retriever import retrieve
from .state import AgentState


def _has_langgraph() -> bool:
    try:
        import langgraph  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def _build_langgraph(llm):
    from langgraph.graph import END, StateGraph
    g = StateGraph(AgentState)
    g.add_node("planner", plan)
    g.add_node("retriever", retrieve)
    g.add_node("executor", lambda s: execute(s, llm))
    g.set_entry_point("planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "executor")
    g.add_edge("executor", END)
    return g.compile()


def answer(question: str, llm=None) -> AgentState:
    """Run the full graph and return the final state."""
    if llm is None:
        from .llm import get_llm
        llm = get_llm()

    state: AgentState = {"question": question, "messages": []}
    if _has_langgraph():
        return _build_langgraph(llm)(state)
    # fallback: same nodes in sequence
    state = plan(state)
    state = retrieve(state)
    state = execute(state, llm)
    return state
