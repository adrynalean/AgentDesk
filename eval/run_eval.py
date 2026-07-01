"""Measure grounded faithfulness: RAG agent vs a no-retrieval baseline.

Persists results to eval/results.json (+ history) and regenerates eval/RESULTS.md
with a resume-ready line. The resume line is flagged safe only when a real
LLM-as-judge was used (keyless mock runs validate the pipeline, not answer quality).

Usage:
    python -m agentdesk.rag.ingest ./corpus     # build the index first
    python -m eval.run_eval eval/dataset_large.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agentdesk.config import settings
from agentdesk.graph import answer as run_agent
from agentdesk.llm import GroundedMockLLM, get_llm
from agentdesk.react import run_react
from agentdesk.rag.store import get_store
from eval.metrics import aggregate, faithfulness, faithfulness_llm

RESULTS_JSON = Path("eval/results.json")
RESULTS_HISTORY = Path("eval/results_history.jsonl")
RESULTS_MD = Path("eval/RESULTS.md")


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(p / 100 * len(s)), len(s) - 1)]


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _model_label(real_llm: bool) -> str:
    if not real_llm:
        return "grounded-mock (keyless)"
    if settings.llm_provider == "openai":
        return f"openai:{settings.openai_model}"
    return f"bedrock:{settings.bedrock_model_id}"


def _resume_line(r: dict) -> str:
    return (
        "Multi-agent RAG (LangGraph, hybrid RRF retrieval, ReAct tool-calling): "
        f"{r['faithfulness_rag']:.0%} faithfulness vs {r['faithfulness_baseline']:.0%} "
        f"no-retrieval baseline on a {r['questions']}-question benchmark over "
        f"{r['chunks']} chunks; p95 latency {r['latency_p95_ms']:.0f}ms."
    )


def write_results(r: dict) -> None:
    RESULTS_JSON.write_text(json.dumps(r, indent=2), encoding="utf-8")
    with RESULTS_HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n")

    safe = r["judge"] == "llm-as-judge"
    banner = ("> ✅ **Résumé-safe** — scored by an LLM judge over model-generated answers."
              if safe else
              "> ⚠️ **Not résumé-safe** — keyless mock (extractive) + lexical judge; this "
              "validates the pipeline, not answer quality. Re-run with `OPENAI_API_KEY` set.")
    RESULTS_MD.write_text(
        f"""# Latest eval results

_Generated {r['timestamp']} · dataset `{r['dataset']}` · model `{r['model']}` · judge `{r['judge']}`_

| Metric | Value |
|--------|-------|
| Corpus chunks | {r['chunks']} |
| Questions | {r['questions']} |
| Faithfulness (RAG) | {r['faithfulness_rag']:.1%} |
| Faithfulness (baseline) | {r['faithfulness_baseline']:.1%} |
| Lift | {r['lift']:.2f}x |
| Latency mean / p95 | {r['latency_mean_ms']:.0f}ms / {r['latency_p95_ms']:.0f}ms |

**Résumé line**
> {_resume_line(r)}

{banner}
""",
        encoding="utf-8",
    )


def main(path: str) -> None:
    rows = load(path)
    llm = get_llm()
    real_llm = not isinstance(llm, GroundedMockLLM)
    judge = (lambda a, c: faithfulness_llm(a, c, llm)) if real_llm else faithfulness
    rag_scores, base_scores, latencies = [], [], []

    for row in rows:
        q = row["question"]
        t0 = time.perf_counter()
        state = run_agent(q, llm=llm)
        latencies.append(time.perf_counter() - t0)
        contexts = [c["text"] for c in state.get("retrieved", [])]
        rag_scores.append(judge(state.get("answer", ""), contexts))
        base_scores.append(judge(run_react(llm, q, contexts=[]).answer, contexts))

    rag, base = aggregate(rag_scores), aggregate(base_scores)
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": path,
        "model": _model_label(real_llm),
        "judge": "llm-as-judge" if real_llm else "lexical",
        "chunks": get_store().count(),
        "questions": rag["n"],
        "faithfulness_rag": rag["mean"],
        "faithfulness_baseline": base["mean"],
        "lift": round(rag["mean"] / base["mean"], 3) if base["mean"] else 0.0,
        "latency_mean_ms": round(sum(latencies) / len(latencies) * 1000, 1),
        "latency_p95_ms": round(_pct(latencies, 95) * 1000, 1),
    }

    print(f"[eval] judge: {results['judge']}   model: {results['model']}")
    print(f"[eval] chunks: {results['chunks']}   questions: {results['questions']}")
    print(f"[eval] faithfulness  RAG: {results['faithfulness_rag']:.1%}   "
          f"baseline: {results['faithfulness_baseline']:.1%}   lift: {results['lift']:.2f}x")
    print(f"[eval] latency  mean: {results['latency_mean_ms']:.0f}ms  "
          f"p95: {results['latency_p95_ms']:.0f}ms")

    write_results(results)
    print(f"[eval] wrote {RESULTS_JSON} and {RESULTS_MD}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/dataset.jsonl")
