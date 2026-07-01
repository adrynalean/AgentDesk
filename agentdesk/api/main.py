"""FastAPI app: a streaming /chat endpoint over the agent graph."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ..graph import answer

app = FastAPI(title="AgentDesk", version="0.0.1")


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """Non-streaming answer (works against the skeleton graph).

    TODO(M4): add a streaming variant with sse-starlette that yields tokens
    from llm.stream_tokens as the executor synthesizes the answer.
    """
    state = answer(req.question)
    return {
        "answer": state.get("answer"),
        "plan": state.get("plan"),
        "retrieved": len(state.get("retrieved", [])),
    }
