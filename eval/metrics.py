"""Evaluation metrics for the RAG assistant."""
from __future__ import annotations


def faithfulness(answer: str, contexts: list[str], judge=None) -> float:
    """Fraction of answer claims supported by retrieved contexts (0..1).

    TODO(M5): use an LLM-as-judge to decompose the answer into atomic claims and
    check each against `contexts`. Return supported_claims / total_claims.
    """
    raise NotImplementedError("wire the judge in M5")


def aggregate(scores: list[float]) -> dict[str, float]:
    """Summary stats over per-question faithfulness scores."""
    if not scores:
        return {"n": 0, "mean": 0.0}
    return {"n": len(scores), "mean": sum(scores) / len(scores)}
