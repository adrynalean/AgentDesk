"""Retriever node: ground the plan in the vector store."""
from __future__ import annotations

from ..rag.retriever import retrieve_chunks
from ..state import AgentState


def retrieve(state: AgentState) -> AgentState:
    """Populate state['retrieved'] with scored, re-ranked chunks.

    TODO(M2): retrieve per plan-step and dedupe across steps.
    """
    chunks = retrieve_chunks(state["question"])
    return {
        **state,
        "retrieved": chunks,
        "messages": [{"role": "retriever", "content": f"{len(chunks)} chunks"}],
    }
