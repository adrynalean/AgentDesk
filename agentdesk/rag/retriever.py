"""Hybrid retrieval: dense vector search + lexical re-ranking."""
from __future__ import annotations

from typing import Any

from ..config import settings


def retrieve_chunks(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Return top-k scored chunks for a query.

    TODO(M1):
        - dense: Chroma similarity search on the query embedding
        - lexical: BM25 / keyword overlap over the candidate pool
        - fuse scores (e.g., reciprocal-rank fusion) and return top-k
    Each item: {"id", "text", "score", "source"}.
    """
    k = k or settings.top_k
    return []  # empty until M1 lands
