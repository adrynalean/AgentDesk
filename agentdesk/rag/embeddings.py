"""Text embeddings.

Prod: sentence-transformers (semantic embeddings). Dev/fallback: a deterministic
hashed character-n-gram embedding — no heavy deps, but real vectors with genuine
cosine similarity, so retrieval actually ranks by relevance out of the box.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from ..config import settings

_DIM = 384  # fallback vector dim


def _stable_hash(s: str) -> int:
    """Process-independent hash (Python's built-in hash() is salted per run)."""
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")


def _ngrams(text: str, n: int = 3) -> list[str]:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    toks = text.split()
    grams = list(toks)  # unigrams
    for w in toks:  # + char trigrams for sub-word overlap
        grams += [w[i:i + n] for i in range(max(len(w) - n + 1, 1))]
    return grams


class Embedder:
    """Unified embedder. Uses sentence-transformers if available, else hashing."""

    def __init__(self) -> None:
        self._st = None
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(settings.embed_model)
            self.dim = self._st.get_sentence_embedding_dimension()
            self.backend = "sentence-transformers"
        except Exception:  # noqa: BLE001
            self.dim = _DIM
            self.backend = "hashing"

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            vecs = np.asarray(self._st.encode(texts, normalize_embeddings=True))
            return vecs.astype(np.float32)
        return np.vstack([self._hash_embed(t) for t in texts])

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for g in _ngrams(text):
            h = _stable_hash(g) % self.dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
