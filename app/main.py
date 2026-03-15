"""FastAPI application for the Cohere + Wikipedia chat service."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.cohere_client import CohereClient
from app.config import get_settings
from app.store import HistoryStore
from app.wikipedia_client import WikipediaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: initialise / tear down shared resources
# ---------------------------------------------------------------------------

cohere_client: CohereClient | None = None
wiki_client: WikipediaClient | None = None
history_store = HistoryStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cohere_client, wiki_client
    settings = get_settings()
    cohere_client = CohereClient(settings.cohere_api_key, settings.cohere_model)
    wiki_client = WikipediaClient()
    logger.info("Clients initialised (model=%s)", settings.cohere_model)
    yield
    await cohere_client.close()
    await wiki_client.close()
    logger.info("Clients closed")


app = FastAPI(title="Cohere Wiki Chat", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")


class ChatResponse(BaseModel):
    response: str


class HistoryRecord(BaseModel):
    id: int
    query: str
    response: str
    created_at: str


# ---------------------------------------------------------------------------
# Wikipedia tool definition (sent to Cohere)
# ---------------------------------------------------------------------------

WIKIPEDIA_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "wikipedia_search",
        "description": (
            "Search Wikipedia for information about a topic. Returns article "
            "titles, snippets, and a summary of the top result. Use this tool "
            "whenever the user asks a factual question that could be answered "
            "using Wikipedia."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on Wikipedia.",
                }
            },
            "required": ["query"],
        },
    },
}

MAX_TOOL_ROUNDS = 5


# ------------------------
# POST /simple_chat (model-only, no tools)
# ------------------------


@app.post("/chat", summary="Ask the model directly", response_model=ChatResponse)
async def simple_chat(req: ChatRequest):
    """Ask the model directly (no Wikipedia/tool calls). Saves to history."""
    assert cohere_client is not None

    messages: List[Dict[str, Any]] = [{"role": "user", "content": req.query}]
    try:
        resp = await cohere_client.chat(messages=messages, tools=None)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    answer = resp.text or ""
    history_store.save(req.query, answer)
    return ChatResponse(response=answer)

# ---------------------------------------------------------------------------
# POST /chat with wikipedia tool use
# ---------------------------------------------------------------------------


@app.post("/wiki_chat", summary="Ask the model with Wikipedia tool use", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Accept a user query, call Cohere (with Wikipedia tool use), return the
    model's final response."""
    assert cohere_client is not None
    assert wiki_client is not None

    messages: List[Dict[str, Any]] = [{"role": "user", "content": req.query}]
    tools = [WIKIPEDIA_TOOL]

    final_text = ""

    for _round in range(MAX_TOOL_ROUNDS):
        try:
            resp = await cohere_client.chat(messages=messages, tools=tools)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        # If the model produced a final answer (not a tool call), we're done.
        if resp.finish_reason != "TOOL_CALL":
            final_text = resp.text
            break

        if not resp.tool_calls:
            # Defensive: finish_reason said TOOL_CALL but no calls present.
            final_text = resp.text
            break

        # Append the assistant's tool-calling turn to the conversation.
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function_name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in resp.tool_calls
            ],
        }
        if resp.raw.get("message", {}).get("tool_plan"):
            assistant_msg["tool_plan"] = resp.raw["message"]["tool_plan"]
        messages.append(assistant_msg)

        # Execute each tool call and feed results back.
        for tc in resp.tool_calls:
            result = await _execute_tool(tc.function_name, tc.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    history_store.save(req.query, final_text)
    return ChatResponse(response=final_text)


async def _execute_tool(name: str, arguments_json: str) -> str:
    """Dispatch a tool call to its local implementation."""
    if name == "wikipedia_search":
        return await _exec_wikipedia_search(arguments_json)
    return json.dumps({"error": f"unknown tool: {name}"})


async def _exec_wikipedia_search(arguments_json: str) -> str:
    """Run a Wikipedia search + summary and return JSON for the model."""
    assert wiki_client is not None

    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"invalid arguments JSON: {exc}"})

    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "empty query argument"})

    logger.info("Wikipedia search: %s", query)

    results = await wiki_client.search(query, limit=3)
    if not results:
        return json.dumps({"results": []})

    # Fetch a richer summary for the top result.
    top = results[0]
    summary = await wiki_client.get_summary(top.title)

    if summary is None:
        # Fall back to search snippets only.
        return json.dumps({"results": [asdict(r) for r in results]})

    payload: Dict[str, Any] = {
        "top_result": asdict(summary),
        "other_results": [asdict(r) for r in results[1:]],
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------


@app.get("/history", summary="Get chat history", response_model=List[HistoryRecord])
async def history():
    """Return every stored query/response pair."""
    records = history_store.all()
    return [
        HistoryRecord(
            id=r.id,
            query=r.query,
            response=r.response,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]
