"""Web-search tool: DuckDuckGo Instant Answer API (degrades gracefully offline)."""
from __future__ import annotations

import requests


def web_search(query: str, k: int = 3) -> str:
    """Return a short factual snippet for a query."""
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            timeout=8,
        )
        data = r.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        topics = [t.get("Text", "") for t in data.get("RelatedTopics", []) if t.get("Text")]
        return " | ".join(topics[:k]) or "no results"
    except Exception as exc:  # noqa: BLE001
        return f"search unavailable: {exc}"


web_search.description = "Search the web and return a short factual answer."
