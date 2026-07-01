"""Run the faithfulness eval over a labeled question set.

Usage:  python -m eval.run_eval eval/dataset.jsonl
Dataset lines: {"question": "...", "answer": "..."}  (answer = reference, optional)
"""
from __future__ import annotations

import json
import sys

from agentdesk.graph import answer as run_agent
from eval.metrics import aggregate, faithfulness


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(path: str) -> None:
    rows = load(path)
    scores: list[float] = []
    for row in rows:
        state = run_agent(row["question"])
        contexts = [c.get("text", "") for c in state.get("retrieved", [])]
        try:
            scores.append(faithfulness(state.get("answer", ""), contexts))
        except NotImplementedError:
            print("[eval] faithfulness() not implemented yet (M5).")
            break
    print(f"[eval] {len(rows)} questions -> {aggregate(scores)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/dataset.jsonl")
