"""Evaluation metrics: grounded-faithfulness of an answer against retrieved context.

If a real LLM judge is configured it can be plugged in; the default lexical judge
needs no API key and produces a real, reproducible score: the fraction of the
answer's content words that are supported by the retrieved context.
"""
from __future__ import annotations

import re

_STOP = set("a an the of to in on for and or is are was were be as at by with from "
            "that this it its into how what when who where why which".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in _STOP and len(w) > 2}


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of the answer's content words supported by the contexts (0..1)."""
    ans = _content_words(answer)
    if not ans:
        return 0.0
    ctx = set()
    for c in contexts:
        ctx |= _content_words(c)
    return round(len(ans & ctx) / len(ans), 3)


def aggregate(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"n": 0, "mean": 0.0}
    return {"n": len(scores), "mean": round(sum(scores) / len(scores), 3)}
