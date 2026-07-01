# AgentDesk — Multi-Agent RAG Assistant

A multi-agent, retrieval-augmented assistant. A **planner** decomposes a request, a
**retriever** grounds it in a vector store (Chroma) with hybrid re-ranking, and a
**tool-executor** runs tools (web search, calculator, code) — orchestrated as a graph in
**LangGraph** with shared memory. Served over a streaming **FastAPI** API (React frontend),
containerized with **Docker**, and measured by an **evaluation harness** that reports
answer faithfulness against a labeled question set.

> Status: **scaffold**. Nodes, tools, and RAG are stubbed with clear `TODO`s. The build
> plan below turns this skeleton into the working system described on the résumé.

## Architecture

```
          ┌──────────┐     ┌───────────┐     ┌──────────────┐
 query ─► │ planner  │ ──► │ retriever │ ──► │ tool-executor│ ──► answer (stream)
          └────┬─────┘     └─────┬─────┘     └──────┬───────┘
               │  shared AgentState (LangGraph)     │
               └──────────── memory ────────────────┘
                                 │
                          Chroma vector store  ◄── rag/ingest.py
```

## Layout

| Path | Purpose |
|------|---------|
| `agentdesk/graph.py` | LangGraph wiring of the three nodes + memory |
| `agentdesk/agents/` | `planner`, `retriever`, `executor` node logic |
| `agentdesk/tools/` | tool implementations + registry for tool-calling |
| `agentdesk/rag/` | Chroma ingestion + hybrid retrieval/re-ranking |
| `agentdesk/llm.py` | model providers (AWS Bedrock / OpenAI) via config |
| `agentdesk/api/main.py` | FastAPI streaming endpoint |
| `eval/` | faithfulness eval harness + metrics |

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env                                # add your keys
python -m agentdesk.rag.ingest ./data              # build the Chroma index
uvicorn agentdesk.api.main:app --reload            # serve
python -m eval.run_eval eval/dataset.jsonl         # measure faithfulness
```

## Build plan (milestones)

- [ ] **M1 — RAG spine:** implement `rag/ingest.py` + `rag/retriever.py` (Chroma, hybrid re-rank).
- [ ] **M2 — Graph:** wire planner → retriever → executor in `graph.py` with `AgentState`.
- [ ] **M3 — Tools:** finish `tools/` (web_search, calculator) + tool-calling loop.
- [ ] **M4 — API:** token-streaming FastAPI endpoint + minimal React client.
- [ ] **M5 — Eval:** faithfulness metric over a labeled set; record the real numbers.

## Target metrics (fill with measured values once M5 runs)

| Metric | Target |
|--------|--------|
| Answer faithfulness (RAG vs no-retrieval baseline) | ≥ 90% vs ~60% |
| Eval set size | 250 questions |
| Retrieval p95 latency | ~2s end-to-end |
