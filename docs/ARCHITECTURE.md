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

### Model Runner (run_batch_generation)

Runs in a ThreadPoolExecutor to avoid blocking the asyncio event
loop. Passes all prompts in the batch to the HuggingFace pipeline
simultaneously. The pipeline handles padding internally.

Outputs are routed back to each request's individual result_queue
via call_soon_threadsafe, which safely bridges the sync executor
thread back to the async event loop.

### SSE Streaming

Each request has its own asyncio.Queue (result_queue). The
submit() generator drains this queue and yields tokens as they
arrive. FastAPI's StreamingResponse forwards these to the client
as Server-Sent Events.

### /metrics Endpoint

Exposes real-time server state: active requests, queue depth,
batch size, total requests processed, total tokens generated,
and KV cache memory estimation based on TinyLlama-1.1B's
architecture constants.

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
collect_batch()
Waits up to 50ms for batch to fill
Returns list of 1-8 Requests
|
▼
run_batch_generation() [in ThreadPoolExecutor]
Builds prompts list
Calls pipeline(prompts, batch_size=N)
GPU processes all N sequences in one forward pass
Routes generated text to each request's result_queue
|
▼
submit() generator
Drains result_queue as tokens arrive
Yields SSE formatted chunks
|
▼
Client receives streaming response

## Known Limitations

**No per-request streaming**
Full response delivered at once rather than token by token.
Fix: implement TextIteratorStreamer per request with token
demultiplexing.

**No KV cache management**
Memory pressure estimated but not actively managed. Under
sustained load with long sequences, GPU memory could fill
without graceful degradation.
Fix: implement block-based KV cache management (see PagedAttention).

**No request timeout or backpressure**
Requests queue indefinitely. Under extreme load, queue depth
grows unbounded.
Fix: add per-request timeout (408) and max queue depth (429/503).

**Single worker**
Hardware constraint — RTX 4050 6GB can only fit one
TinyLlama pipeline instance.
Fix: larger GPU allows multiple pipeline instances for true
parallel execution.
