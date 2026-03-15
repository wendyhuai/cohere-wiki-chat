"""Async client for searching and summarising Wikipedia articles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"
REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"
USER_AGENT = "cohere-wiki-chat/1.0 (https://github.com/wendyhuai/cohere-wiki-chat)"


@dataclass
class SearchResult:
    title: str
    snippet: str


@dataclass
class Summary:
    title: str
    description: str
    extract: str


class WikipediaClient:
    """Async client for the Wikipedia search and summary APIs."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=15.0, headers={"User-Agent": USER_AGENT}
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def search(self, query: str, limit: int = 3) -> List[SearchResult]:
        """Search Wikipedia articles matching *query*."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "format": "json",
        }
        resp = await self._http.get(MEDIAWIKI_API, params=params)
        resp.raise_for_status()
        data = resp.json()

        results: List[SearchResult] = []
        for item in data.get("query", {}).get("search", []):
            results.append(
                SearchResult(title=item["title"], snippet=item.get("snippet", ""))
            )
        return results

    async def get_summary(self, title: str) -> Optional[Summary]:
        """Fetch the plain-text summary of a Wikipedia article by title."""
        encoded = quote(title, safe="")
        resp = await self._http.get(f"{REST_SUMMARY}/{encoded}")
        if resp.status_code != 200:
            logger.warning(
                "Wikipedia summary returned %d for %r", resp.status_code, title
            )
            return None
        data = resp.json()
        return Summary(
            title=data.get("title", ""),
            description=data.get("description", ""),
            extract=data.get("extract", ""),
        )
