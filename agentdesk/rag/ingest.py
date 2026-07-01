"""Ingest documents into a Chroma collection.

Usage:  python -m agentdesk.rag.ingest ./data
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..config import settings


def chunk(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Naive fixed-size character chunker with overlap (swap for token-aware in M1)."""
    step = size - overlap
    return [text[i : i + size] for i in range(0, max(len(text), 1), step)]


def ingest(data_dir: str) -> int:
    """Read files under data_dir, chunk, embed, and upsert into Chroma.

    TODO(M1):
        - client = chromadb.PersistentClient(path=settings.chroma_dir)
        - col = client.get_or_create_collection("agentdesk")
        - embed with sentence-transformers(settings.embed_model)
        - col.upsert(ids, documents, embeddings, metadatas)
    Returns the number of chunks indexed.
    """
    files = [p for p in Path(data_dir).rglob("*") if p.is_file()]
    total = sum(len(chunk(p.read_text(errors="ignore"))) for p in files)
    print(f"[ingest] would index ~{total} chunks from {len(files)} files "
          f"into {settings.chroma_dir} (TODO: M1)")
    return total


if __name__ == "__main__":
    ingest(sys.argv[1] if len(sys.argv) > 1 else "./data")
