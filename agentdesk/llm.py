"""Model-provider abstraction: OpenAI or AWS Bedrock, chosen via config."""
from __future__ import annotations

from typing import Iterator

from .config import settings


def get_chat_model():
    """Return a LangChain chat model for the configured provider.

    TODO(M2): return the concrete client and share it across nodes.
    """
    if settings.llm_provider == "openai":
        # from langchain_openai import ChatOpenAI
        # return ChatOpenAI(model=settings.openai_model, streaming=True)
        raise NotImplementedError("wire ChatOpenAI in M2")
    if settings.llm_provider == "bedrock":
        # from langchain_aws import ChatBedrock
        # return ChatBedrock(model_id=settings.bedrock_model_id, region=settings.aws_region)
        raise NotImplementedError("wire ChatBedrock in M2")
    raise ValueError(f"unknown provider: {settings.llm_provider}")


def stream_tokens(prompt: str) -> Iterator[str]:
    """Yield answer tokens for the streaming API.

    TODO(M4): replace with real token streaming from get_chat_model().
    """
    raise NotImplementedError("wire streaming in M4")
