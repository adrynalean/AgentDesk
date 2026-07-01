# Retrieval in AgentDesk

Retrieval is hybrid. AgentDesk combines dense vector similarity with lexical
overlap using reciprocal-rank fusion, so a chunk that is both semantically and
lexically close to the query is ranked higher.

Embeddings are produced by sentence-transformers when available. In development,
AgentDesk uses a deterministic hashed character-n-gram embedding so that vectors
are stable across runs and retrieval still ranks by relevance.

Documents are split into overlapping character chunks of about 800 characters
with 120 characters of overlap before they are embedded and indexed.

The retriever returns the top matching chunks, and the executor grounds its final
answer only in those retrieved chunks.
