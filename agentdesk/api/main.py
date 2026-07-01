"""FastAPI app: ingest docs, then chat over the multi-agent RAG graph."""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..graph import answer
from ..rag.ingest import ingest

app = FastAPI(title="AgentDesk", version="0.1.0")


@lru_cache
def _llm():
    from ..llm import get_llm
    return get_llm()


class ChatRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    data_dir: str = "./data"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest_docs(req: IngestRequest) -> dict:
    return {"chunks_indexed": ingest(req.data_dir)}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    state = answer(req.question, llm=_llm())
    return {
        "answer": state.get("answer"),
        "plan": state.get("plan"),
        "sources": [c.get("source") for c in state.get("retrieved", [])],
        "tool_calls": state.get("tool_calls", []),
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    state = answer(req.question, llm=_llm())
    text = state.get("answer", "")

    def gen():
        for word in text.split():
            yield f"data: {word} \n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
