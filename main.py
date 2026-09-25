from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import asyncio
import torch
import uvicorn
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time
import traceback

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter
)

# OpenTelemetry setup
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

NUM_WORKERS = 1  

# Higher = better GPU utilization, high memory usage, but more latency for small batches
BATCH_SIZE = 8

# Higher = more time waiting to batch up, better GPU utilization, but higher latency
BATCH_WAIT_TIME = 0.05

# TinyLlama-1.1B architecture constants for KV cache estimation
KV_LAYERS = 22
KV_HEADS = 32
KV_HEAD_DIM = 64
KV_BYTES_PER_VALUE = 2  # bfloat16

MAX_QUEUE_DEPTH = 20      # reject with 503 if queue exceeds this
REQUEST_TIMEOUT_S = 30    # reject with 408 if waiting longer than this
MAX_NEW_TOKENS = 200

class ChatRequest(BaseModel):
    question: str

@dataclass
class Request:
    id: str
    prompt: str
    result_queue: asyncio.Queue

@dataclass
class ServerStats:
    active_requests: int = 0
    total_requests_processed: int = 0
    total_tokens_generated: int = 0
    queue_depth: int = 0
    current_batch_size: int = 0
    last_batch_time: float = 0.0


# one executor thread per worker
executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)


@dataclass
class Slot:
    """One in-flight request. past is that row's KV only, with no padding."""

    request: Request
    past: tuple
    next_id: int
    generated: list
    printed: str = ""
    new_count: int = 0
    finished: bool = False


def _legacy_layers(past):
    """(key, value) per layer. Transformers 5.17 DynamicCache yields a third item."""
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    if hasattr(past, "layers"):
        return tuple((layer.keys, layer.values) for layer in past.layers)
    return tuple((item[0], item[1]) for item in past)


def _as_model_cache(legacy):
    from transformers.cache_utils import DynamicCache
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(legacy)
    return DynamicCache(ddp_cache_data=legacy)


def _slice_row(legacy, index, length):
    """Keep one batch row and the first `length` real (right-padded) positions."""
    layers = []
    for key, value in legacy:
        layers.append((
            key[index:index + 1, :, :length, :].contiguous(),
            value[index:index + 1, :, :length, :].contiguous(),
        ))
    return tuple(layers)


def _take_real_suffix(legacy, index, length):
    """After a left-padded decode step, keep the last `length` real positions."""
    layers = []
    for key, value in legacy:
        layers.append((
            key[index:index + 1, :, -length:, :].contiguous(),
            value[index:index + 1, :, -length:, :].contiguous(),
        ))
    return tuple(layers)


def _batch_left_pad(legacies):
    max_len = max(leg[0][0].shape[2] for leg in legacies)
    n_layers = len(legacies[0])
    batched = []
    for layer_i in range(n_layers):
        keys = []
        values = []
        for leg in legacies:
            key, value = leg[layer_i]
            pad = max_len - key.shape[2]
            if pad:
                zeros = torch.zeros(
                    key.shape[0], key.shape[1], pad, key.shape[3],
                    dtype=key.dtype, device=key.device,
                )
                key = torch.cat([zeros, key], dim=2)
                value = torch.cat([zeros, value], dim=2)
            keys.append(key)
            values.append(value)
        batched.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
    return tuple(batched)

class InferenceEngine:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.queue = asyncio.Queue()
        self.stats = ServerStats()
        
        print("Loading pipeline...")
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Pipeline loaded.")
        
        
    async def collect_batch(self):
        """
        wait for first req (blocking)
        collect more requests till time
        return batch upto batch_size requests
        """
        first = await self.queue.get()
        batch = [first]
        
        # wait for more requests to batch up
        deadline = asyncio.get_event_loop().time() + BATCH_WAIT_TIME
        
        while len(batch) < BATCH_SIZE:
            timeout = deadline - asyncio.get_event_loop().time()
            if timeout <= 0:
                break
            try:
                req = await asyncio.wait_for(self.queue.get(), timeout)
                batch.append(req)
            except asyncio.TimeoutError:
                break
            
        print(f"Collected batch of {len(batch)} requests")
        return batch

    def _chat_prompt(self, request):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request.prompt},
        ]
        return self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _emit(self, slot, token_id, loop):
        tokenizer = self.pipe.tokenizer
        eos = tokenizer.eos_token_id
        if token_id == eos or slot.new_count >= MAX_NEW_TOKENS:
            slot.finished = True
            loop.call_soon_threadsafe(slot.request.result_queue.put_nowait, None)
            return
        slot.generated.append(token_id)
        slot.new_count += 1
        decoded = tokenizer.decode(slot.generated, skip_special_tokens=True)
        if not decoded.endswith("\ufffd"):
            delta = decoded[len(slot.printed):]
            slot.printed = decoded
            if delta:
                loop.call_soon_threadsafe(
                    slot.request.result_queue.put_nowait, delta
                )
        slot.next_id = token_id
        if slot.new_count >= MAX_NEW_TOKENS:
            slot.finished = True
            loop.call_soon_threadsafe(slot.request.result_queue.put_nowait, None)

    def _prefill(self, requests, loop):
        tokenizer = self.pipe.tokenizer
        model = self.pipe.model
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = next(model.parameters()).device
        prompts = [self._chat_prompt(request) for request in requests]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            out = model(**encoded, use_cache=True)
        legacy = _legacy_layers(out.past_key_values)
        slots = []
        for i, request in enumerate(requests):
            length = int(encoded["attention_mask"][i].sum().item())
            past = _slice_row(legacy, i, length)
            token_id = int(out.logits[i, length - 1].argmax().item())
            slot = Slot(request=request, past=past, next_id=token_id, generated=[])
            self._emit(slot, token_id, loop)
            slots.append(slot)
        return slots

    def _decode_step(self, slots, loop):
        model = self.pipe.model
        device = next(model.parameters()).device
        running = [slot for slot in slots if not slot.finished]
        if not running:
            return
        lengths = [slot.past[0][0].shape[2] for slot in running]
        max_len = max(lengths)
        legacy = _batch_left_pad([slot.past for slot in running])
        input_ids = torch.tensor(
            [[slot.next_id] for slot in running], dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            len(running), max_len + 1, dtype=torch.long, device=device
        )
        position_ids = torch.zeros(
            len(running), 1, dtype=torch.long, device=device
        )
        for i, length in enumerate(lengths):
            attention_mask[i, max_len - length:] = 1
            position_ids[i, 0] = length
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=_as_model_cache(legacy),
                use_cache=True,
            )
        new_legacy = _legacy_layers(out.past_key_values)
        for i, slot in enumerate(running):
            slot.past = _take_real_suffix(new_legacy, i, lengths[i] + 1)
            token_id = int(out.logits[i, -1].argmax().item())
            self._emit(slot, token_id, loop)

    def _fail(self, requests, error):
        for request in requests:
            request.result_queue.put_nowait(f"Error: {error}")
            request.result_queue.put_nowait(None)

    def _close_finished(self, slots):
        still = []
        for slot in slots:
            if slot.finished:
                self.stats.active_requests -= 1
                self.stats.total_requests_processed += 1
                self.stats.total_tokens_generated += slot.new_count
            else:
                still.append(slot)
        self.stats.current_batch_size = len(still)
        return still

    async def worker(self):
        print("Worker started")
        loop = asyncio.get_event_loop()
        active = []

        while True:
            incoming = []
            if not active:
                incoming = await self.collect_batch()
            else:
                while len(active) < BATCH_SIZE:
                    try:
                        incoming.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            if incoming:
                self.stats.queue_depth = self.queue.qsize()
                self.stats.active_requests += len(incoming)
                self.stats.current_batch_size = len(active) + len(incoming)
                print(f"Admitting {len(incoming)} requests, {len(active)} already running")
                try:
                    newcomers = await loop.run_in_executor(
                        executor, lambda reqs=incoming: self._prefill(reqs, loop)
                    )
                    active.extend(newcomers)
                except Exception as e:
                    traceback.print_exc()
                    self._fail(incoming, e)
                    self.stats.active_requests -= len(incoming)
                finally:
                    for _ in incoming:
                        self.queue.task_done()
                active = self._close_finished(active)

            if not active:
                continue

            try:
                await loop.run_in_executor(
                    executor, lambda slots=list(active): self._decode_step(slots, loop)
                )
            except Exception as e:
                traceback.print_exc()
                self._fail([slot.request for slot in active], e)
                self.stats.active_requests -= len(active)
                active = []
                self.stats.current_batch_size = 0
                continue
            active = self._close_finished(active)
                



    def _update_after_batch(self, batch_size, tokens, duration):
        self.stats.active_requests -= batch_size
        self.stats.total_requests_processed += batch_size
        self.stats.total_tokens_generated += tokens
        self.stats.last_batch_time = duration
        self.stats.current_batch_size = 0  
        
    async def submit(self, prompt: str, request_id: str):
        
        if self.queue.qsize() >= MAX_QUEUE_DEPTH:
            raise HTTPException(status_code=503,
            detail=f"Server overloaded. Queue depth: {self.queue.qsize()}")
        
        result_queue = asyncio.Queue()
        request = Request(
            id=request_id,
            prompt=prompt,
            result_queue=result_queue
        )
        await self.queue.put(request)
        try:
            async def generate_with_timeout():
                while True:
                    token = await asyncio.wait_for(
                        result_queue.get(),
                        timeout=REQUEST_TIMEOUT_S
                    )
                    if token is None:
                        break
                    yield token

            async for token in generate_with_timeout():
                yield token

        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail="Request timed out waiting for generation"
            )

    def get_kv_cache_stats(self) -> dict:
        """
        Estimate KV cache memory usage.
        
        Formula for TinyLlama-1.1B:
        - 22 transformer layers
        - 32 attention heads  
        - 64 head dimension
        - 2 matrices per layer (K and V)
        - 2 bytes per value (bfloat16)
        
        bytes_per_token = 2 * 22 * 32 * 64 * 2 = 180,224 bytes ≈ 180KB
        """
        bytes_per_token = KV_BYTES_PER_VALUE * KV_LAYERS * KV_HEADS * KV_HEAD_DIM * 2
        
        # estimate tokens in flight: active_requests * avg_tokens_so_far
        # use 100 as rough midpoint of 200 max tokens
        estimated_tokens_in_flight = self.stats.active_requests * 100
        estimated_kv_bytes = estimated_tokens_in_flight * bytes_per_token
        
        gpu_total = 0
        gpu_used = 0
        
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            gpu_used = torch.cuda.memory_allocated(0)
            
        return {
            "bytes_per_token": bytes_per_token,
            "estimated_tokens_in_flight": estimated_tokens_in_flight,
            "estimated_kv_bytes": estimated_kv_bytes,
            "estimated_kv_mb": round(estimated_kv_bytes / (1024**2)),
            "gpu_total_mb": round(gpu_total / (1024**2)),
            "gpu_used_mb": round(gpu_used / (1024**2)),
            "gpu_free_mb": round((gpu_total - gpu_used) / (1024**2)) if gpu_total else 0,
            "gpu_utilisation_percent": round(gpu_used / gpu_total * 100,2) if gpu_total else 0
        }
        
    
app = FastAPI()
engine = InferenceEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def chat_sse(question: str):
    async for token in engine.submit(question, "default_id"):
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(engine.worker())

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate")
async def main(payload: ChatRequest):
    
    with tracer.start_as_current_span("http_request") as span:
        span.set_attribute("prompt.length", len(payload.question))
    return StreamingResponse(
        chat_sse(payload.question),
        media_type="text/event-stream"
    )

@app.get("/metrics")
async def metrics():
    kv_stats = engine.get_kv_cache_stats()
    return {
        "server": {
            "active_requests": engine.stats.active_requests,
            "queue_depth": engine.queue.qsize(),
            "queue_capacity": MAX_QUEUE_DEPTH,
            "queue_utilization_pct": round(
                engine.queue.qsize() / MAX_QUEUE_DEPTH * 100, 1
            ),
            "current_batch_size": engine.stats.current_batch_size,
            "total_requests_processed": engine.stats.total_requests_processed,
            "total_tokens_generated": engine.stats.total_tokens_generated,
            "last_batch_time_s": round(engine.stats.last_batch_time, 3),
        },
        "kv_cache": kv_stats,
        "config": {
            "batch_size": BATCH_SIZE,
            "batch_wait_ms": BATCH_WAIT_TIME * 1000,
            "max_queue_depth": MAX_QUEUE_DEPTH,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "max_new_tokens": 200,
            "model": "TinyLlama-1.1B-Chat-v1.0",
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)