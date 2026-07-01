"""Web-search tool stub (wire to a real search API in M3)."""
from __future__ import annotations


def web_search(query: str, k: int = 3) -> list[dict]:
    """Return up to k search results as {title, url, snippet}.

    TODO(M3): back this with a real provider (Tavily / SerpAPI / Bing).
    """
    raise NotImplementedError("wire a search provider in M3")


web_search.description = "Search the web and return top results."
