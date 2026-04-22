from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import asyncio
import torch
from transformers import pipeline, TextIteratorStreamer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

NUM_WORKERS = 1  
BATCH_SIZE = 8
BATCH_WAIT_TIME = 0.05

class ChatRequest(BaseModel):
    question: str

@dataclass
class Request:
    id: str
    prompt: str
    result_queue: asyncio.Queue

# one executor thread per worker
executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

class InferenceEngine:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.queue = asyncio.Queue()
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
                      # no inner thread — run pipeline directly in executor
                    # streamer yields tokens as they generate
                    # import threading
                    # gen = threading.Thread(
                    #     target=self.pipe,
                    #     kwargs=generation_kwargs
                    # )
                    # gen.start()
                    # for token in streamer:
                    #     loop.call_soon_threadsafe(
                    #         request.result_queue.put_nowait, token
                    #     )
                    # gen.join()
                    # loop.call_soon_threadsafe(
                    #     request.result_queue.put_nowait, None
                    # )
                    print(f"Running GPU batch: {len(prompts)} prompts")
                    outputs = self.pipe(prompts, max_new_tokens=200, return_full_text=False, batch_size=len(prompts),pad_token_id=self.pipe.tokenizer.eos_token_id) 
                    
                    for request, output in zip(batch, outputs):
                        generated_text = output[0]["generated_text"]
                        loop.call_soon_threadsafe(
                            request.result_queue.put_nowait, generated_text
                        )
                        loop.call_soon_threadsafe(
                            request.result_queue.put_nowait, None
                        )
                    print(f"Batch complete")


                # fire into executor — worker immediately free for next request
                # asyncio.ensure_future(
                #     loop.run_in_executor(executor, run_generation)
                # )
                await loop.run_in_executor(executor, run_batch_generation)

            except Exception as e:
                print(f"Error: {e}")
                for request in batch:
                    request.result_queue.put_nowait(f"Error: {e}")
                    request.result_queue.put_nowait(None)
            finally:
                for _ in batch:
                    self.queue.task_done()

        
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