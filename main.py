# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

from dataclasses import dataclass
import asyncio
import torch
from transformers import pipeline, TextIteratorStreamer
from threading import Thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.responses import StreamingResponse

class ChatRequest(BaseModel):
    question: str
# Request model
@dataclass
class Request:
    id: str
    prompt : str
    result_queue: asyncio.Queue
    
class InferenceEngine:
    def __init__(self, model_name:str  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.queue = asyncio.Queue()  
        self.pipe =   pipeline("text-generation", model=model_name, dtype=torch.bfloat16, device_map="auto", )
        
    async def worker(self):
        """ Worker loop: get requests -> generate -> push tokens to result queue  -> push None sentinel to signal end of generation """
        while True:
            request:Request = await self.queue.get()    
            print(f"Processing request {request.id} with prompt {request.prompt}")  
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a friendly chatbot who always responds in the style of a pirate",
                    },
                    {"role": "user", "content":   request.prompt},
                ]
                streamer = TextIteratorStreamer(self.pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
                prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.pipe.tokenizer(prompt, return_tensors="pt").to(self.pipe.device)
                generation_kwargs = dict(text_inputs=prompt, streamer=streamer, max_new_tokens=200)
                # New thread for generation
                thread = Thread(target=self.pipe, kwargs=generation_kwargs)
                thread.start()
                # Push tokens to result queue 
                for token in streamer:
                    print(f"Token :{token}")
                    await request.result_queue.put(token)
                thread.join()               
                # Signal end of generation
                await request.result_queue.put(None)     
            except Exception as e:
                await request.result_queue.put(f"Error: {str(e)}")
                await request.result_queue.put(None)
                print(f"Error processing request {request.id}: {e}")      
            finally:
                self.queue.task_done()
                
    async def submit(self, prompt:str, request_id:str, ):
        """ Submit request and yield tokens from the result queue """
        result_queue = asyncio.Queue()    
        request = Request(id=request_id, prompt=prompt, result_queue=result_queue)
        await self.queue.put(request)        
        while True:
            token = await result_queue.get()
            if token is None:
                break
            yield token


app = FastAPI()
engine = InferenceEngine()

# Allow CORS 
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost:5173",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#SSE - Server-Sent Events 
async def chat_sse(question: str):
    async for new_token in engine.submit(question, "default_id"):
        yield f"data: {new_token}\n\n"
    yield "data: [DONE]\n\n"

@app.on_event("startup")
async def startup_event():
    # Start the inference engine worker
    asyncio.create_task(engine.worker())
    
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def main(payload: ChatRequest):
    return StreamingResponse(chat_sse(payload.question), media_type="text/event-stream") 
    


