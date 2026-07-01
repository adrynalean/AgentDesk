"""End-to-end tests using the dev fallbacks (numpy store + grounded mock LLM).

Uses a temp CHROMA_DIR so tests don't touch a real index.
"""
import os
import tempfile

os.environ["CHROMA_DIR"] = tempfile.mkdtemp()

from agentdesk.graph import answer            # noqa: E402
from agentdesk.rag.ingest import ingest       # noqa: E402
from agentdesk.tools import calculator        # noqa: E402
from eval.metrics import faithfulness         # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _seed():
    return ingest(DATA)


def test_calculator_tool():
    assert calculator("42*(3+1)") == "168"


def test_ingest_indexes_chunks():
    assert _seed() >= 3


def test_rag_answer_is_grounded():
    _seed()
    state = answer("What is the default vector store?")
    assert "chroma" in state["answer"].lower()     # correct, retrieved fact
    assert state["retrieved"]                       # retrieval happened
    # answer is well supported by the retrieved context
    ctx = [c["text"] for c in state["retrieved"]]
    assert faithfulness(state["answer"], ctx) >= 0.6


def test_planner_decomposes_compound_question():
    state = answer("What is the default vector store and what orchestrates the agents?")
    assert len(state["plan"]) >= 2                  # split into sub-queries


def test_tool_calling_path():
    _seed()
    state = answer("What is 42 * (3 + 1)?")
    assert state["answer"] == "168"
    assert any(tc["action"] == "calculator" for tc in state["tool_calls"])
