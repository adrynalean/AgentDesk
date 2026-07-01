# AgentDesk — Multi-Agent RAG Assistant

![CI](https://github.com/adrynalean/AgentDesk/actions/workflows/ci.yml/badge.svg)

A multi-agent, retrieval-augmented assistant. A **planner** decomposes a request,
a **retriever** grounds each sub-query in a vector store (hybrid dense + lexical),
and a **tool-executor** runs tools (calculator, web search) via a **ReAct** loop and
synthesizes an answer grounded in the retrieved context — orchestrated as a graph in
**LangGraph** with shared memory, served over **FastAPI**, and measured by an
**evaluation harness** that reports grounded faithfulness.

It runs **out of the box with no API key**: a deterministic hashed-embedding vector
store and a context-grounded mock LLM stand in for Chroma / sentence-transformers /
an LLM provider, which are used automatically when installed.

## Architecture

```
question ─► planner ─► retriever ─► executor ─► grounded answer
             (sub-      (hybrid RAG:  (ReAct: tools +
              queries)   dense+lexical) context synthesis)
                     shared AgentState (LangGraph, + sequential fallback)
                                   │
                    vector store: Chroma  ·  dev: numpy cosine store
                    embeddings:  sentence-transformers  ·  dev: hashed n-grams
```

| Path | Role |
|------|------|
| `agentdesk/agents/{planner,retriever,executor}.py` | the three agent nodes |
| `agentdesk/graph.py` | LangGraph wiring (+ dependency-free fallback) |
| `agentdesk/react.py` | ReAct reason→act→observe loop |
| `agentdesk/rag/{embeddings,store,ingest,retriever}.py` | embeddings, vector store, hybrid retrieval |
| `agentdesk/tools/` | tool registry (calculator, web_search) |
| `agentdesk/llm.py` | OpenAI / Bedrock providers + grounded mock |
| `agentdesk/api/main.py` | FastAPI `/ingest`, `/chat`, `/chat/stream` + serves the UI |
| `frontend/` | buildless React chat UI (plan, tool calls, sources) |
| `eval/` | grounded-faithfulness harness (RAG vs no-retrieval baseline) |
| `scripts/` | corpus + eval-set builders (stdlib docs, offline) |

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows
pip install -r requirements.txt
python -m agentdesk.rag.ingest ./data                # index the sample corpus
python -m eval.run_eval eval/dataset.jsonl           # measure faithfulness
uvicorn agentdesk.api.main:app --reload              # serve the API
pytest tests/                                        # 5 tests, green without any key
```

Example:

```bash
$ python -m agentdesk.rag.ingest ./data
[ingest] indexed 6 chunks from 3 files (embedder: hashing)

$ python -m eval.run_eval eval/dataset.jsonl
[eval] questions: 8
[eval] faithfulness  RAG: 87.5%   baseline(no-context): 0.0%
```

Ask it something:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question":"What orchestrates the agents?"}'
# {"answer":"AgentDesk is orchestrated with LangGraph.", "sources":["agentdesk.md", ...]}
```

## Chat UI

`uvicorn agentdesk.api.main:app` also serves a **React chat UI** at `http://localhost:8000/` —
it shows the final answer plus the agent's plan, tool calls (with inputs/observations),
and retrieval sources for every message. No build step needed (React via ESM).

## Benchmark

See [BENCHMARK.md](BENCHMARK.md) and [eval/RESULTS.md](eval/RESULTS.md) — a 927-chunk
corpus built from Python stdlib docs, a seeded 250-question set, and a harness that
reports faithfulness + latency (LLM-as-judge when a provider key is configured).

## Going to production

Set `OPENAI_API_KEY` (or configure Bedrock) in `.env` and `pip install` the full
`requirements.txt` — AgentDesk then uses **Chroma**, **sentence-transformers**, and a
real **LLM** automatically, with no code changes. Point ingestion at your own corpus
and re-run `eval/run_eval.py` to record faithfulness on your data.

## Status

Core pipeline is implemented and tested: multi-agent graph, hybrid RAG, ReAct
tool-calling, streaming API, and a working eval harness. Next: swap the sample
corpus for a real one and record faithfulness with a live LLM judge.
