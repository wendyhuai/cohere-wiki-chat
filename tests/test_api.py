"""Tests for the API endpoints with mocked external services."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.cohere_client import ChatResponse, ToolCall
from app.main import app, history_store
from app.store import HistoryStore


@pytest.fixture()
def client():
    """Return a TestClient with mocked lifespan dependencies.

    We enter the TestClient context first (which runs the lifespan and creates
    real clients), then overwrite the module-level globals with mocks so that
    all requests go through the fakes.
    """
    from app import main

    with TestClient(app, raise_server_exceptions=False) as tc:
        # Overwrite globals *after* lifespan has run.
        mock_cohere = AsyncMock()
        mock_wiki = AsyncMock()

        main.cohere_client = mock_cohere
        main.wiki_client = mock_wiki
        main.history_store = HistoryStore()

        yield tc, mock_cohere, mock_wiki

    # Restore
    main.cohere_client = None
    main.wiki_client = None


def _simple_chat_response(text: str) -> ChatResponse:
    return ChatResponse(
        finish_reason="COMPLETE",
        text=text,
        tool_calls=[],
        raw={
            "message": {"content": [{"type": "text", "text": text}]},
            "finish_reason": "COMPLETE",
        },
    )


def _tool_call_response(tool_call_id: str, name: str, args: dict) -> ChatResponse:
    return ChatResponse(
        finish_reason="TOOL_CALL",
        text="",
        tool_calls=[
            ToolCall(
                id=tool_call_id,
                function_name=name,
                arguments=json.dumps(args),
            )
        ],
        raw={
            "message": {
                "tool_plan": "I will search Wikipedia.",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args),
                        },
                    }
                ],
            },
            "finish_reason": "TOOL_CALL",
        },
    )


# ── POST /chat ──────────────────────────────────────────────────────────


class TestChatEndpoint:
    def test_simple_query(self, client):
        tc, mock_cohere, _mock_wiki = client
        mock_cohere.chat.return_value = _simple_chat_response("Buzz Aldrin.")

        resp = tc.post("/chat", json={"query": "Who walked on the moon?"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "Buzz Aldrin."

    def test_empty_query_rejected(self, client):
        tc, _mock_cohere, _mock_wiki = client
        resp = tc.post("/chat", json={"query": ""})
        assert resp.status_code == 422  # Pydantic validation

    def test_missing_query_rejected(self, client):
        tc, _mock_cohere, _mock_wiki = client
        resp = tc.post("/chat", json={})
        assert resp.status_code == 422

    def test_tool_call_flow(self, client):
        """Simulate: model requests wikipedia_search → we execute → model replies."""
        tc, mock_cohere, mock_wiki = client

        # First call: model asks for a tool call.
        # Second call: model gives final answer.
        mock_cohere.chat.side_effect = [
            _tool_call_response("tc1", "wikipedia_search", {"query": "Moon landing"}),
            _simple_chat_response("Buzz Aldrin was second."),
        ]

        # Mock Wikipedia responses.
        from app.wikipedia_client import SearchResult, Summary

        mock_wiki.search.return_value = [
            SearchResult(title="Apollo 11", snippet="First Moon landing")
        ]
        mock_wiki.get_summary.return_value = Summary(
            title="Apollo 11",
            description="First crewed Moon landing",
            extract="Apollo 11 was the first mission to land humans on the Moon.",
        )

        resp = tc.post(
            "/chat", json={"query": "Who was the second person to walk on the moon?"}
        )
        assert resp.status_code == 200
        assert "Buzz Aldrin" in resp.json()["response"]

        # Cohere should have been called twice.
        assert mock_cohere.chat.call_count == 2

    def test_cohere_error_returns_502(self, client):
        tc, mock_cohere, _mock_wiki = client
        mock_cohere.chat.side_effect = RuntimeError("API down")

        resp = tc.post("/chat", json={"query": "test"})
        assert resp.status_code == 502

    def test_chat_saves_to_history(self, client):
        tc, mock_cohere, _mock_wiki = client
        mock_cohere.chat.return_value = _simple_chat_response("42")

        tc.post("/chat", json={"query": "answer"})
        resp = tc.get("/history")
        records = resp.json()
        assert len(records) == 1
        assert records[0]["query"] == "answer"
        assert records[0]["response"] == "42"


# ── GET /history ────────────────────────────────────────────────────────


class TestHistoryEndpoint:
    def test_empty_history(self, client):
        tc, _mock_cohere, _mock_wiki = client
        resp = tc.get("/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_multiple_records(self, client):
        tc, mock_cohere, _mock_wiki = client
        mock_cohere.chat.side_effect = [
            _simple_chat_response("A1"),
            _simple_chat_response("A2"),
        ]

        tc.post("/chat", json={"query": "Q1"})
        tc.post("/chat", json={"query": "Q2"})

        resp = tc.get("/history")
        records = resp.json()
        assert len(records) == 2
        assert records[0]["query"] == "Q1"
        assert records[1]["query"] == "Q2"
