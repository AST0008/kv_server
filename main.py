from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import asyncio
import torch
import uvicorn
from transformers import pipeline, TextIteratorStreamer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time

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

    async def worker(self):
        print("Worker started")
        loop = asyncio.get_event_loop()

        while True:
            # request: Request = await self.queue.get()
            batch = await self.collect_batch()
            
            print(f"Processing batch of {len(batch)} requests")
            # print(f"Processing {request.id} ")
            
            # update stats 
            self.stats.queue_depth = self.queue.qsize()
            self.stats.current_batch_size = len(batch)
            self.stats.active_requests += len(batch)
            
            try:
                prompts = []
                
                
                for request in batch:
                    messages  = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": request.prompt}
                    ]
                    
                    prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)

                def run_batch_generation(prompts=prompts,batch = batch):
                    
                    total_tokens = 0
                    batch_start = time.time()
                    
                    print(f"Running GPU batch: {len(prompts)} prompts")
                    outputs = self.pipe(prompts, max_new_tokens=200, return_full_text=False, batch_size=len(prompts),pad_token_id=self.pipe.tokenizer.eos_token_id) 
                    
                    for request, output in zip(batch, outputs):
                        generated_text = output[0]["generated_text"]
                        
                        token_count = len(self.pipe.tokenizer(generated_text,add_special_tokens=False).input_ids)
                        total_tokens += token_count
                        
                        loop.call_soon_threadsafe(
                            request.result_queue.put_nowait, generated_text
                        )
                        loop.call_soon_threadsafe(
                            request.result_queue.put_nowait, None
                        )
                        
                    loop.call_soon_threadsafe(
                        self._update_after_batch, len(batch), total_tokens,time.time() - batch_start
                    )
                    
                    print(f"Batch complete")

                await loop.run_in_executor(executor, run_batch_generation)

            except Exception as e:
                print(f"Error: {e}")
                for request in batch:
                    request.result_queue.put_nowait(f"Error: {e}")
                    request.result_queue.put_nowait(None)
                self.stats.active_requests -= len(batch)   
            finally:
                for _ in batch:
                    self.queue.task_done()

    def _update_after_batch(self, batch_size, tokens, duration):
        self.stats.active_requests -= batch_size
        self.stats.total_requests_processed += batch_size
        self.stats.total_tokens_generated += tokens
        self.stats.last_batch_time = duration
        self.stats.current_batch_size = 0  
        
    async def submit(self, prompt: str, request_id: str):
        result_queue = asyncio.Queue()
        request = Request(
            id=request_id,
            prompt=prompt,
            result_queue=result_queue
        )
        await self.queue.put(request)
        while True:
            token = await result_queue.get()
            if token is None:
                break
            yield token

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
    return StreamingResponse(
        chat_sse(payload.question),
        media_type="text/event-stream"
    )

@app.get("/metrics")
async def get_metrics():
    kv_stats = engine.get_kv_cache_stats()
    return {
        "server": {
            "active_requests": engine.stats.active_requests,
            "queue_depth": engine.queue.qsize(),
            "current_batch_size": engine.stats.current_batch_size,
            "total_requests_processed": engine.stats.total_requests_processed,
            "total_tokens_generated": engine.stats.total_tokens_generated,
            "last_batch_time_s": round(engine.stats.last_batch_time, 3),
        },
        "kv_cache": kv_stats,
        "config": {
            "batch_size": BATCH_SIZE,
            "batch_wait_ms": BATCH_WAIT_TIME * 1000,
            "max_new_tokens": 200,
            "model": "TinyLlama-1.1B-Chat-v1.0"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)