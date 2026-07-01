"""Retriever node: ground each plan sub-query, dedupe across steps."""
from __future__ import annotations

from ..config import settings
from ..rag.retriever import retrieve_chunks
from ..state import AgentState


def retrieve(state: AgentState) -> AgentState:
    seen: set[str] = set()
    merged: list[dict] = []
    for sub in state.get("plan", [state["question"]]):
        for hit in retrieve_chunks(sub, settings.top_k):
            key = hit.get("id") or hit["text"][:60]
            if key not in seen:
                seen.add(key)
                merged.append(hit)
    merged = merged[: settings.top_k]
    return {
        **state,
        "retrieved": merged,
        "messages": [{"role": "retriever", "content": f"{len(merged)} chunks"}],
    }
