# Architecture Overview

## 1) Components

### Backend (`main.py`)

- **FastAPI app** exposes:
  - `GET /` health-ish route
  - `POST /generate` chat route
- **Chat request model** (`ChatRequest`) validates JSON payload (`question: str`).
- **LLM pipeline** is created once at startup using Hugging Face `transformers.pipeline`.
- **Streaming generator**:
  - `chat(question)` yields generated text token chunks.
  - `chat_sse(question)` wraps chunks as SSE frames (`data: ...\n\n`) and emits `[DONE]`.
- **CORS middleware** allows local frontend origins.

### Frontend (`frontend/src/App.tsx`)

- **Input state** stores user question text.
- **Request state** tracks streaming state and errors.
- **SSE parser** reads `fetch(...).body` stream, splits by SSE event boundary (`\n\n`), extracts `data:` lines, and appends content to response state.
- **UI** shows a prompt form and live response area.

## 2) Connection Flow

1. User enters prompt in React UI and submits form.
2. Frontend sends `POST http://localhost:8000/generate` with JSON body.
3. FastAPI validates body, calls `chat_sse(question)`.
4. Backend model generates text incrementally; each chunk is emitted as SSE `data:` event.
5. Frontend stream reader receives chunks and updates response text live.
6. Backend sends `data: [DONE]` to signal completion.

## 3) Runtime Boundaries

- **Browser boundary**: React app in Vite dev server (`:5173`).
- **API boundary**: FastAPI app (`:8000`).
- **Model boundary**: Transformers pipeline running in backend Python process.

## 4) Current Tradeoffs (Basic)

- Single-process backend; no queue or worker isolation.
- In-memory request handling only; no persistence layer.
- Frontend uses manual SSE parsing with `fetch` stream (works well for POST body use-case).
- Local CORS allowlist is development-focused.

## 5) Possible Next Steps

- Add request cancellation support (`AbortController`) in frontend.
- Add auth/rate limits on backend.
- Add observability (request IDs, latency metrics, structured logs).
- Move model serving behind a dedicated inference service if scale grows.
