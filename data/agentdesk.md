# AgentDesk Overview

AgentDesk is a multi-agent retrieval-augmented assistant. It answers questions by
combining three cooperating agents with a vector store and a set of tools.

The three agents are the planner, the retriever, and the executor. The planner
decomposes a question into focused sub-queries. The retriever grounds each
sub-query in a vector store. The executor runs tools when needed and synthesizes
a final answer that is grounded in the retrieved context.

AgentDesk is orchestrated with LangGraph. Each agent is a node in a state graph,
and a shared AgentState object is threaded between the nodes. When LangGraph is
not installed, AgentDesk falls back to running the same nodes in sequence.

The default vector store is Chroma. In development, AgentDesk falls back to a
numpy cosine-similarity store persisted to JSON, so retrieval works without any
external services.
