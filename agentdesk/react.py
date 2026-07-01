"""ReAct loop: reason -> act (tool-call) -> observe -> grounded answer.

Dependency-free and unit-testable. The graph's executor node uses this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .prompts import build_prompt
from .tools import TOOLS

_ACTION = re.compile(r"Action:\s*(.+)")
_INPUT = re.compile(r"Action Input:\s*(.+)")
_FINAL = re.compile(r"Final Answer:\s*(.+)", re.S)


@dataclass
class Step:
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


@dataclass
class Result:
    answer: str
    steps: list[Step] = field(default_factory=list)


def _parse(text: str):
    fin = _FINAL.search(text)
    if fin:
        return fin.group(1).strip(), None, None
    act = _ACTION.search(text)
    inp = _INPUT.search(text)
    return None, (act.group(1).strip() if act else None), (inp.group(1).strip() if inp else None)


def run_react(llm, question: str, contexts: list[dict[str, Any]] | None = None,
              max_steps: int = 4) -> Result:
    contexts = contexts or []
    scratch, steps = "", []
    for _ in range(max_steps):
        turn = llm.generate(build_prompt(question, contexts, scratch))
        final, action, action_input = _parse(turn)
        if final is not None:
            return Result(final, steps)
        obs = (TOOLS[action].run(action_input or "") if action in TOOLS
               else f"error: unknown tool '{action}'")
        steps.append(Step(action, action_input, obs))
        scratch += f"{turn}\nObservation: {obs}\n"
    return Result("(stopped: max steps reached)", steps)
