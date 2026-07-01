"""Auto-generate a QA eval set from the stdlib corpus.

For each documented object, creates a definition question whose reference answer
is the first sentence of its docstring. Samples N questions (seeded, reproducible).

Usage:  python scripts/build_eval.py [n] [out.jsonl]
"""
from __future__ import annotations

import importlib
import inspect
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow run from repo root
from build_corpus import MODULES  # reuse the same module list  # noqa: E402


def _first_sentence(doc: str) -> str:
    doc = inspect.cleandoc(doc).strip().replace("\n", " ")
    m = re.match(r"(.+?[.!?])(\s|$)", doc)
    return (m.group(1) if m else doc)[:300]


def collect() -> list[dict]:
    items: list[dict] = []
    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:  # noqa: BLE001
            continue
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj) or inspect.isbuiltin(obj)):
                continue
            if getattr(obj, "__module__", modname) != modname:
                continue
            doc = inspect.getdoc(obj)
            if not doc or len(doc) < 20:
                continue
            items.append({
                "question": f"What does {modname}.{name} do?",
                "answer": _first_sentence(doc),
                "source": f"{modname.replace('.', '_')}.md",
            })
    return items


def build(n: int = 250, out: str = "eval/dataset_large.jsonl") -> int:
    items = collect()
    random.seed(42)
    random.shuffle(items)
    picked = items[:n]
    with open(out, "w", encoding="utf-8") as fh:
        for it in picked:
            fh.write(json.dumps(it) + "\n")
    print(f"[eval] wrote {len(picked)} questions to {out} (pool: {len(items)})")
    return len(picked)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    build(n, sys.argv[2] if len(sys.argv) > 2 else "eval/dataset_large.jsonl")
