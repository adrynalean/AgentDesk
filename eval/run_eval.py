"""Measure grounded faithfulness: RAG agent vs a no-retrieval baseline.

Usage:
    python -m agentdesk.rag.ingest ./data      # build the index first
    python -m eval.run_eval eval/dataset.jsonl
"""
from __future__ import annotations

import json
import sys
import time

from agentdesk.graph import answer as run_agent
from agentdesk.llm import get_llm
from agentdesk.react import run_react
from agentdesk.llm import GroundedMockLLM
from agentdesk.rag.store import get_store
from eval.metrics import aggregate, faithfulness, faithfulness_llm


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(p / 100 * len(s)), len(s) - 1)]


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(path: str) -> None:
    rows = load(path)
    llm = get_llm()
    # LLM-as-judge when a real provider is configured; lexical judge otherwise.
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

        # baseline: answer WITHOUT retrieval, scored against the same contexts
        base_answer = run_react(llm, q, contexts=[]).answer
        base_scores.append(judge(base_answer, contexts))

    print(f"[eval] judge: {'LLM-as-judge' if real_llm else 'lexical (keyless mock)'}")

    rag, base = aggregate(rag_scores), aggregate(base_scores)
    print(f"[eval] corpus chunks indexed: {get_store().count()}")
    print(f"[eval] questions: {rag['n']}")
    print(f"[eval] faithfulness  RAG: {rag['mean']:.1%}   baseline(no-context): {base['mean']:.1%}")
    if base["mean"]:
        print(f"[eval] lift: {rag['mean'] / base['mean']:.2f}x")
    print(f"[eval] latency  mean: {sum(latencies)/len(latencies)*1000:.0f}ms  "
          f"p95: {_pct(latencies, 95)*1000:.0f}ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/dataset.jsonl")
