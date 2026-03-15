"""Thin async client for the Cohere v2 Chat API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

COHERE_CHAT_URL = "https://api.cohere.com/v2/chat"


@dataclass
class ToolCall:
    id: str
    function_name: str
    arguments: str  # JSON string


@dataclass
class ChatResponse:
    finish_reason: str
    text: str
    tool_calls: List[ToolCall]
    raw: Dict[str, Any]


class CohereClient:
    """Async client for the Cohere v2 Chat API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._http = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        """Send a non-streaming chat request and return a parsed response."""
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        resp = await self._http.post(COHERE_CHAT_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            body = resp.text
            logger.error("Cohere API returned %d: %s", resp.status_code, body)
            raise RuntimeError(f"Cohere API returned {resp.status_code}: {body}")

        data = resp.json()
        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> ChatResponse:
        msg = data.get("message", {})
        finish_reason = data.get("finish_reason", "")

        # Extract text content.
        text_parts = []
        for part in msg.get("content", []) or []:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        text = "".join(text_parts)

        # Extract tool calls.
        tool_calls = []
        for tc in msg.get("tool_calls", []) or []:
            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    function_name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
            )

        return ChatResponse(
            finish_reason=finish_reason,
            text=text,
            tool_calls=tool_calls,
            raw=data,
        )
