"""Hybrid retrieval: dense vector search + lexical re-ranking (reciprocal-rank fusion)."""
from __future__ import annotations

import re
from typing import Any

from ..config import settings
from .store import embed_query, get_store


def _lexical_scores(query: str, hits: list[dict]) -> list[float]:
    terms = set(re.findall(r"\w+", query.lower()))
    scores = []
    for h in hits:
        words = re.findall(r"\w+", h["text"].lower())
        scores.append(sum(w in terms for w in words) / (len(words) or 1))
    return scores


def retrieve_chunks(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Return top-k chunks fusing dense similarity with lexical overlap.

    Reciprocal-rank fusion over the dense ranking and the lexical ranking gives a
    small but real boost to chunks that are both semantically and lexically close.
    """
    k = k or settings.top_k
    pool = get_store().query(embed_query(query), max(k * 3, k))
    if not pool:
        return []

    dense_rank = {h["id"] if "id" in h else i: r
                  for r, h in enumerate(pool) for i in [id(h)]}
    lex = _lexical_scores(query, pool)
    lex_order = sorted(range(len(pool)), key=lambda i: -lex[i])
    lex_rank = {i: r for r, i in enumerate(lex_order)}

    fused = []
    for i, h in enumerate(pool):
        rrf = 1 / (60 + i) + 1 / (60 + lex_rank[i])   # reciprocal-rank fusion
        fused.append({**h, "score": round(float(h.get("score", 0)), 4), "rrf": rrf})
    fused.sort(key=lambda x: -x["rrf"])
    return fused[:k]
