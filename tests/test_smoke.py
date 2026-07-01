"""Smoke tests: the skeleton must import and run end-to-end (stubbed)."""
from agentdesk.graph import answer
from agentdesk.tools import calculator


def test_graph_runs_end_to_end():
    state = answer("what is this project?")
    assert "answer" in state
    assert state["plan"]              # planner produced steps
    assert "messages" in state


def test_calculator_tool_works():
    assert calculator("3 * (4 + 1)") == "15"
