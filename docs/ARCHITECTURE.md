# Architecture

## Overview

A minimal LLM inference server built from scratch to understand
the core systems challenges in production model serving.

## Components

### InferenceEngine

The core class. Owns the request queue, the model pipeline,
and the batch scheduler. Single instance shared across all
FastAPI request handlers.

### Request Queue (asyncio.Queue)

Incoming HTTP requests are converted to Request objects and
placed in an asyncio.Queue. This decouples HTTP handling from
GPU execution — FastAPI can accept new connections while the
GPU is busy with the current batch.

### Batch Scheduler (collect_batch)

Waits for the first request, then collects additional requests
for up to 50ms or until batch_size=8 is reached. Flushes the
batch to the GPU regardless of fill level when the timer expires.

Tradeoff: longer wait = fuller batches = better GPU utilization
but higher latency. 50ms is a reasonable default for interactive
use cases.

### Model Runner (prefill / decode)

Runs in a ThreadPoolExecutor to avoid blocking the asyncio event
loop. TinyLlama-1.1B runs through HuggingFace. KV lives in 16-token
pages. Each decode step gathers those pages into one contiguous
cache, runs ordinary attention, and writes back only the new token.
HuggingFace does not read the pages.

Outputs are routed back to each request's individual result_queue
via call_soon_threadsafe, which safely bridges the sync executor
thread back to the async event loop.

### SSE Streaming

Each request has its own asyncio.Queue (result_queue). The
submit() generator drains this queue and yields tokens as they
arrive. FastAPI's StreamingResponse forwards these to the client
as Server-Sent Events.

### /metrics Endpoint

Exposes the page pool: `used_blocks`, `free_blocks`, `filled_tokens`,
and `kv_mb`, plus CUDA memory. It does not estimate KV as active
requests times 100 tokens.

## Request Lifecycle

POST /generate
|
▼
FastAPI handler
Creates Request(id, prompt, result_queue)
Adds to engine.queue
Returns StreamingResponse
|
▼
reserve()
Takes 16-token pages before the stream starts
503 before any SSE byte if the free list cannot cover the prompt
Second 503 if the queue is already at 20; those pages are released
|
▼
Scheduler
GPU idle: wait 50ms, cap 8
Already decoding: take a waiting request if a seat is free
|
▼
Prefill / decode [in ThreadPoolExecutor]
Gathers pages into one contiguous cache for HuggingFace
Routes each token to that request's result_queue
|
▼
submit() generator
Drains result_queue as tokens arrive
Yields SSE formatted chunks
|
▼
Client receives streaming response

## Known Limitations

**Attention still reads a contiguous copy**
Each decode step gathers pages into one cache for HuggingFace
and writes back only the new token. The attention kernel does
not read the pages.

**Backpressure**
Admission returns 503 before any SSE byte when free pages cannot
cover the prompt. Queue depth is a second 503. A token wait past
30s returns 408.

**Single worker**
Hardware constraint — RTX 4050 6GB can only fit one
TinyLlama pipeline instance.
Fix: larger GPU allows multiple pipeline instances for true
parallel execution.
