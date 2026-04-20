from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import asyncio
import torch
from transformers import pipeline, TextIteratorStreamer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

NUM_WORKERS = 1  # RTX 4050 6GB — only one pipeline fits safely

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

    async def worker(self):
        print("Worker started")
        loop = asyncio.get_event_loop()

        while True:
            request: Request = await self.queue.get()
            print(f"Processing {request.id}")
            try:
                messages = [
                    {"role": "system", "content": "You are a pirate chatbot."},
                    {"role": "user", "content": request.prompt},
                ]
                streamer = TextIteratorStreamer(
                    self.pipe.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                prompt = self.pipe.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                generation_kwargs = dict(
                    text_inputs=prompt,
                    streamer=streamer,
                    max_new_tokens=200,
                    return_full_text=False
                )

                def run_generation():
                    # no inner thread — run pipeline directly in executor
                    # streamer yields tokens as they generate
                    import threading
                    gen = threading.Thread(
                        target=self.pipe,
                        kwargs=generation_kwargs
                    )
                    gen.start()
                    for token in streamer:
                        loop.call_soon_threadsafe(
                            request.result_queue.put_nowait, token
                        )
                    gen.join()
                    loop.call_soon_threadsafe(
                        request.result_queue.put_nowait, None
                    )

                # fire into executor — worker immediately free for next request
                # asyncio.ensure_future(
                #     loop.run_in_executor(executor, run_generation)
                # )
                await loop.run_in_executor(executor, run_generation)

            except Exception as e:
                print(f"Error: {e}")
                request.result_queue.put_nowait(None)
            finally:
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