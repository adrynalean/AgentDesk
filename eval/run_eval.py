"""Measure grounded faithfulness: RAG agent vs a no-retrieval baseline.

Usage:
    python -m agentdesk.rag.ingest ./data      # build the index first
    python -m eval.run_eval eval/dataset.jsonl
"""
from __future__ import annotations

import json
import sys

from agentdesk.graph import answer as run_agent
from agentdesk.llm import get_llm
from agentdesk.react import run_react
from eval.metrics import aggregate, faithfulness


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(path: str) -> None:
    rows = load(path)
    llm = get_llm()
    rag_scores, base_scores = [], []

    for row in rows:
        q = row["question"]
        state = run_agent(q, llm=llm)
        contexts = [c["text"] for c in state.get("retrieved", [])]
        rag_scores.append(faithfulness(state.get("answer", ""), contexts))

        # baseline: answer WITHOUT retrieval, scored against the same contexts
        base_answer = run_react(llm, q, contexts=[]).answer
        base_scores.append(faithfulness(base_answer, contexts))

    rag, base = aggregate(rag_scores), aggregate(base_scores)
    print(f"[eval] questions: {rag['n']}")
    print(f"[eval] faithfulness  RAG: {rag['mean']:.1%}   baseline(no-context): {base['mean']:.1%}")
    if base["mean"]:
        print(f"[eval] lift: {rag['mean'] / base['mean']:.2f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/dataset.jsonl")
