# cohere-wiki-chat

A REST API that answers user questions using the Cohere Chat API, augmented with a Wikipedia tool for factual lookups. It also exposes a history endpoint for retrieving past queries and responses.

## Getting started

### Prerequisites

- Python 3.9+
- A Cohere API key (sign up at <https://dashboard.cohere.com/welcome/login>)

### Setup

```bash
# Clone the repository
git clone https://github.com/wendyhuai/cohere-wiki-chat.git
cd cohere-wiki-chat

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your Cohere API key
export COHERE_API_KEY="your-key-here"

# (Optional) override the default model and port
export COHERE_MODEL="command-r7b-12-2024"
export PORT=8080
```

### Running the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Using the API

**Chat (with Wikipedia tool use):**

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Who was the second person to walk on the moon?"}'
```

**Query history:**

```bash
curl http://localhost:8080/history
```

### Running tests

```bash
pytest tests/ -v
```

Tests mock both the Cohere API and Wikipedia, so no API key or network access is needed.

## API reference

| Method | Path       | Description                                    |
|--------|------------|------------------------------------------------|
| POST   | `/chat`    | Send a user query; returns the model's answer. |
| GET    | `/history` | Returns all past query/response pairs.         |

### POST /chat

**Request body:**
```json
{ "query": "Who was the second person to walk on the moon?" }
```

**Response (200):**
```json
{ "response": "Buzz Aldrin was the second person to walk on the moon..." }
```

### GET /history

**Response (200):**
```json
[
  {
    "id": 1,
    "query": "Who was the second person to walk on the moon?",
    "response": "Buzz Aldrin was the second person...",
    "created_at": "2025-01-15T12:34:56.789000+00:00"
  }
]
```

## Design decisions

### Architecture

- **FastAPI** was chosen for the web framework because it provides async support out of the box (important for non-blocking calls to Cohere and Wikipedia), automatic OpenAPI documentation, and Pydantic-based request validation.
- **httpx** is used as the HTTP client for both Cohere and Wikipedia because it supports async/await natively, unlike `requests`.
- The codebase is split into focused modules (`cohere_client`, `wikipedia_client`, `store`, `config`, `main`) to keep responsibilities clear and make each component independently testable.

### Tool-calling loop

The `/chat` endpoint implements a multi-turn tool-calling loop: it sends the user query to Cohere with a `wikipedia_search` tool definition. If Cohere responds with `finish_reason: "TOOL_CALL"`, the server executes the Wikipedia search locally, feeds the results back as a `tool` message, and calls Cohere again. This continues for up to 5 rounds (a safety cap) until the model produces a final text response.

When executing a Wikipedia tool call, we first search using the MediaWiki action API (`action=query&list=search`) and then fetch a richer plain-text summary from the REST API (`/api/rest_v1/page/summary/{title}`) for the top result. This gives the model both breadth (multiple search hits) and depth (a full summary) to work with.

### In-memory history store

Query history is stored in a simple thread-safe, in-memory list. This is intentionally minimal: it avoids the need for a database dependency for a take-home exercise. The store is a plain class with a lock, making it easy to swap in a real database later.

### Limitations

- **History is not persistent.** All data is lost when the server restarts. There is no database.
- **No authentication or authorisation.** Any caller can access all endpoints and see the full history.
- **No pagination** on the `/history` endpoint. With enough queries, the response could grow unbounded.
- **Single-process only.** The in-memory store is not shared across workers. Running multiple uvicorn workers would give each its own independent history.
- **Wikipedia-only tool.** The model can only search Wikipedia. It cannot access other sources, perform calculations, or execute code.
- **No streaming.** Responses are returned only after the full Cohere generation completes, which can be slow for long answers or multi-round tool calls.
- **Error handling is basic.** Upstream failures (Cohere API errors, Wikipedia downtime) return generic 502 responses without retries or circuit-breaking.

## What I would change before exposing this to customers

1. **Persistent storage.** Replace the in-memory store with a database (e.g. PostgreSQL) for durable history that survives restarts and scales across workers.
2. **Authentication and rate limiting.** Add API key or OAuth-based auth so that history is scoped per user, and rate-limit requests to prevent abuse and control Cohere API costs.
3. **Pagination and filtering.** The `/history` endpoint should support pagination (`?page=1&per_page=20`), date range filters, and search.
4. **Streaming responses.** Use Cohere's streaming mode and SSE to send partial tokens to the client in real time, improving perceived latency.
5. **Observability.** Add structured logging, request tracing (e.g. OpenTelemetry), and metrics (request latency, error rates, tool call frequency) for production monitoring.
6. **Retries and resilience.** Add retry logic with exponential backoff for transient failures from Cohere and Wikipedia, plus circuit breakers to degrade gracefully.
7. **Input validation and safety.** Enforce max query length, sanitise inputs, and consider content moderation on both inputs and outputs.
8. **Configuration management.** Use a secrets manager (e.g. AWS Secrets Manager, Vault) for the Cohere API key instead of environment variables.
9. **Containerisation and CI/CD.** Add a Dockerfile and CI pipeline for automated testing, linting, and deployment.
10. **Response caching.** Cache Wikipedia results (they change infrequently) to reduce latency and external API calls.

## Resources used

- **Claude Code** (Anthropic's CLI tool) 
- **Cohere v2 Chat API reference** — <https://docs.cohere.com/v2/reference/chat>
- **Wikipedia MediaWiki Action API** — <https://www.mediawiki.org/wiki/API:Search>
- **Wikipedia REST API** (page summaries) — <https://en.wikipedia.org/api/rest_v1/>
- **FastAPI documentation** — <https://fastapi.tiangolo.com/>
- **httpx documentation** — <https://www.python-httpx.org/>
