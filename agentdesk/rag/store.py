"""Vector store.

Prod: Chroma (persistent). Dev/fallback: a numpy cosine store persisted to JSON.
Both expose the same add()/query() interface.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import settings
from .embeddings import get_embedder


class _NumpyStore:
    """Simple persistent cosine-similarity store (dev / no-Chroma)."""

    def __init__(self, path: str) -> None:
        self.path = Path(path) / "store.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ids: list[str] = []
        self.docs: list[str] = []
        self.metas: list[dict] = []
        self.vecs: list[list[float]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            d = json.loads(self.path.read_text())
            self.ids, self.docs = d["ids"], d["docs"]
            self.metas, self.vecs = d["metas"], d["vecs"]

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            {"ids": self.ids, "docs": self.docs, "metas": self.metas, "vecs": self.vecs}))

    def add(self, ids, docs, metas, vecs) -> None:
        existing = set(self.ids)
        for i, doc, meta, vec in zip(ids, docs, metas, vecs):
            if i in existing:
                continue
            self.ids.append(i)
            self.docs.append(doc)
            self.metas.append(meta)
            self.vecs.append([float(x) for x in vec])
        self._save()

    def query(self, vec, k: int) -> list[dict[str, Any]]:
        if not self.vecs:
            return []
        mat = np.asarray(self.vecs, dtype=np.float32)
        q = np.asarray(vec, dtype=np.float32)
        sims = mat @ q / ((np.linalg.norm(mat, axis=1) * np.linalg.norm(q)) + 1e-9)
        top = np.argsort(-sims)[:k]
        return [{"id": self.ids[i], "text": self.docs[i],
                 "source": self.metas[i].get("source"), "score": float(sims[i])}
                for i in top]

    def count(self) -> int:
        return len(self.ids)


class _ChromaStore:
    """Chroma-backed store (prod)."""

    def __init__(self, path: str) -> None:
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection("agentdesk")

    def add(self, ids, docs, metas, vecs) -> None:
        self.col.upsert(ids=list(ids), documents=list(docs),
                        metadatas=list(metas), embeddings=[list(v) for v in vecs])

    def query(self, vec, k: int) -> list[dict[str, Any]]:
        res = self.col.query(query_embeddings=[list(vec)], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [{"text": d, "source": m.get("source"), "score": 1 - dist}
                for d, m, dist in zip(docs, metas, dists)]

    def count(self) -> int:
        return self.col.count()


_store = None


def get_store():
    global _store
    if _store is not None:
        return _store
    try:
        import chromadb  # noqa: F401
        _store = _ChromaStore(settings.chroma_dir)
    except ModuleNotFoundError:
        _store = _NumpyStore(settings.chroma_dir)
    return _store


def embed_query(text: str):
    return get_embedder().embed([text])[0]
