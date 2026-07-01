"""Ingest documents into the vector store.

Usage:  python -m agentdesk.rag.ingest ./data
"""
from __future__ import annotations

import sys
from pathlib import Path

from .embeddings import get_embedder
from .store import get_store

_EXTS = {".md", ".txt", ".rst"}


def chunk(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    step = size - overlap
    out = [text[i:i + size] for i in range(0, max(len(text), 1), step)]
    return [c.strip() for c in out if c.strip()]


def ingest(data_dir: str) -> int:
    """Chunk, embed, and upsert every text file under data_dir. Returns #chunks."""
    files = [p for p in Path(data_dir).rglob("*") if p.suffix.lower() in _EXTS]
    ids, docs, metas = [], [], []
    for p in files:
        for i, ch in enumerate(chunk(p.read_text(encoding="utf-8", errors="ignore"))):
            ids.append(f"{p.name}:{i}")
            docs.append(ch)
            metas.append({"source": p.name})
    if not docs:
        print(f"[ingest] no documents found under {data_dir}")
        return 0
    vecs = get_embedder().embed(docs)
    get_store().add(ids, docs, metas, vecs)
    print(f"[ingest] indexed {len(docs)} chunks from {len(files)} files "
          f"(embedder: {get_embedder().backend})")
    return len(docs)


if __name__ == "__main__":
    ingest(sys.argv[1] if len(sys.argv) > 1 else "./data")
