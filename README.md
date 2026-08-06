# Auto-Agent-X

Auto-Agent-X is a full-stack RAG question-answering prototype for exploring and
explaining code. It combines a FastAPI backend, intent-aware prompt routing,
hybrid retrieval, Server-Sent Events (SSE), and a React chat interface.

> **Project status:** the retrieval layer currently uses deterministic mock
> retrievers. It is useful for developing the orchestration and UI flows, but a
> real index/vector store must be connected before production use.

## Features

- Intent classification for code search, explanation, review, bug fixing, and chat
- Hybrid retrieval interface with citations
- Streaming responses over SSE
- React + TypeScript chat UI with a Vite development proxy
- Structured application and thought-process logging

## Architecture

```text
Auto-Agent-X/
├── backend/
│   ├── api/              # FastAPI routes
│   ├── agents/           # Agent routing and collaboration
│   ├── core/             # Settings and logging
│   ├── infrastructure/   # LLM provider clients
│   ├── models/           # API schemas
│   └── services/         # Intent, retrieval, and RAG services
├── frontend/             # React + Vite application
├── tests/                # Backend API regression tests
├── .env.example          # Safe configuration template
└── requirements.txt      # Python dependencies
```

## Quick start

### Backend

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the API from the repository root:

```bash
python -m uvicorn backend.main:app --reload
```

The health endpoint is available at `http://localhost:8000/health`.

### Frontend

Node.js 20.19+ is required by the current Vite toolchain.

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` requests to the backend at
`http://localhost:8000`.

## API

- `POST /api/v1/chat/` — JSON response
- `POST /api/v1/chat/stream` — SSE response
- `GET /health` — service health check

Example non-streaming request:

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where is the retrieval logic implemented?"}'
```

## Development checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
cd frontend
npm run lint
npm run build
```

Keep credentials in `.env`; only the placeholder `.env.example` belongs in Git.
