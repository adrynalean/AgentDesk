"""Prompt construction for the ReAct executor."""
from __future__ import annotations

from .tools import tool_descriptions

REACT_TEMPLATE = """You are AgentDesk, a careful assistant. Answer using the \
retrieved context and, when needed, tools. Ground every claim in the context.

Available tools:
{tools}

Format:
Thought: reasoning
Action: <tool name>
Action Input: <tool input>
Observation: <result provided to you>
... repeat as needed ...
Thought: I now know the answer
Final Answer: <answer grounded in the context>

Retrieved context:
{context}

Question: {question}
{scratchpad}"""


def build_prompt(question: str, contexts: list[dict], scratchpad: str = "") -> str:
    ctx = "\n".join(f"[{c.get('source', 'doc')}] {c['text']}" for c in contexts) or "(none)"
    return REACT_TEMPLATE.format(
        tools=tool_descriptions(), context=ctx, question=question, scratchpad=scratchpad,
    )
