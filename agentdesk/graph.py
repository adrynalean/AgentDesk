"""LangGraph wiring: planner -> retriever -> tool-executor, with shared memory.

This is the orchestration spine. Each node is a pure function `AgentState -> AgentState`
so the graph stays testable. See agents/ for node logic.
"""
from __future__ import annotations

from .state import AgentState


def build_graph():
    """Compile and return the runnable agent graph.

    TODO(M2): implement with langgraph.graph.StateGraph:
        g = StateGraph(AgentState)
        g.add_node("planner", plan)
        g.add_node("retriever", retrieve)
        g.add_node("executor", execute)
        g.set_entry_point("planner")
        g.add_edge("planner", "retriever")
        g.add_edge("retriever", "executor")
        g.add_edge("executor", END)
        return g.compile()
    """
    from .agents.planner import plan
    from .agents.retriever import retrieve
    from .agents.executor import execute

    # Placeholder linear runner so the skeleton is callable end-to-end.
    def run(state: AgentState) -> AgentState:
        state = plan(state)
        state = retrieve(state)
        state = execute(state)
        return state

    return run


def answer(question: str) -> AgentState:
    graph = build_graph()
    return graph({"question": question, "messages": []})
