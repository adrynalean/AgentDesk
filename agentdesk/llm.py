"""LLM providers (OpenAI / AWS Bedrock) + a grounded mock for dev/eval.

GroundedMockLLM implements the ReAct protocol *and* answers factual questions by
extracting the best sentence from the retrieved context in the prompt — so the
pipeline produces real, grounded answers (and real eval numbers) with no API key.
"""
from __future__ import annotations

import re
from typing import Protocol

from .config import settings

_ARITH = re.compile(r"[\d(][\d\s+\-*/().]*[\d)]")


def _extract_arith(question: str) -> str | None:
    m = _ARITH.search(question)
    if not m:
        return None
    expr = m.group(0).strip()
    if not any(op in expr for op in "+-*/"):
        return None
    expr += ")" * (expr.count("(") - expr.count(")"))   # balance parens
    return expr


class LLM(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str: ...


class OpenAILLM:
    def __init__(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        r = self._client.chat.completions.create(
            model=self._model, max_tokens=max_new_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()


class BedrockLLM:
    def __init__(self) -> None:
        import boto3
        self._client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self._model = settings.bedrock_model_id

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        import json
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = self._client.invoke_model(modelId=self._model, body=body)
        return json.loads(resp["body"].read())["content"][0]["text"].strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _match(term: str, word: str) -> bool:
    """Exact, or a shared 5+ char prefix (so 'orchestrates' ~ 'orchestrated')."""
    if term == word:
        return True
    n = min(len(term), len(word))
    return n >= 5 and term[:5] == word[:5]


_STOP = set("the what who how why when where a an of to in on for and or is are "
            "was were be as at by with from that this it its into does do".split())


def _best_sentence(context: str, question: str) -> str:
    terms = [w for w in re.findall(r"\w+", question.lower())
             if len(w) > 2 and w not in _STOP]
    best, best_score = "", -1
    for s in _sentences(context):
        words = re.findall(r"\w+", s.lower())
        score = sum(any(_match(t, w) for w in words) for t in terms)
        if score > best_score:
            best, best_score = s, score
    return best or "I don't have enough context to answer."


class GroundedMockLLM:
    """Deterministic, context-grounded stand-in (no API key needed)."""

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        question = _after(prompt, "Question:")
        context = _between(prompt, "Retrieved context:", "Question:")
        acted = prompt.count("Observation:") >= 2   # template has one placeholder

        arith = _extract_arith(question)
        if arith and not acted:
            return (f"Thought: this needs arithmetic.\n"
                    f"Action: calculator\nAction Input: {arith}")
        if arith and acted:
            obs = _last(prompt, "Observation:")
            return f"Thought: I have the result.\nFinal Answer: {obs}"
        return f"Thought: answer from context.\nFinal Answer: {_best_sentence(context, question)}"


# ── prompt slicing helpers ───────────────────────────────────────────────────
def _after(text: str, marker: str) -> str:
    idx = text.find(marker)
    return text[idx + len(marker):].split("\n")[0].strip() if idx >= 0 else ""


def _between(text: str, a: str, b: str) -> str:
    i = text.find(a)
    if i < 0:
        return ""
    j = text.find(b, i)
    return text[i + len(a): j if j >= 0 else None].strip()


def _last(text: str, marker: str) -> str:
    return text.rsplit(marker, 1)[-1].split("\n")[0].strip()


def get_llm() -> LLM:
    """Factory: real provider by config, else the grounded mock."""
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAILLM()
    if settings.llm_provider == "bedrock":
        try:
            return BedrockLLM()
        except Exception:  # noqa: BLE001
            pass
    return GroundedMockLLM()
