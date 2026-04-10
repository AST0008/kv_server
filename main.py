# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import time
from xml.parsers.expat import model

# from aiohttp import streamer

import torch
from transformers import pipeline, TextIteratorStreamer
from threading import Thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from fastapi.responses import StreamingResponse


class ChatRequest(BaseModel):
    question: str

app = FastAPI()

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




prompts  = [
    "What is the meaning of life?",
    "What is the airspeed of a laden swallow?",
    "What is the answer to the Ultimate Question of Life, The Universe, and Everything?",
        "What is the capital of Assyria?",
  "Who is hte best football player in the world?",
  "What is the best way to learn Python?",
  "What is the best way to learn machine learning?",
  "What is the best way to learn deep learning?",
  "What is the best way to learn natural language processing?",
  "What is the best way to learn computer vision?",
  "What is the best way to learn reinforcement learning?",
]

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.bfloat16, device_map="auto", )

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content":     "What is the meaning of life?"},
]

streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


inputs = pipe.tokenizer(prompt, return_tensors="pt").to(pipe.device)

generation_kwargs = dict(text_inputs=messages, streamer=streamer, max_new_tokens=200)
thread = Thread(target=pipe, kwargs=generation_kwargs)
thread.start()
for new_token in streamer:
        # print(new_token, end="", flush=True)
        pass
thread.join()

async def chat(question: str):
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content":     question},
    ]
    streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pipe.tokenizer(prompt, return_tensors="pt").to(pipe.device)
    generation_kwargs = dict(text_inputs=messages, streamer=streamer, max_new_tokens=200)
    thread = Thread(target=pipe, kwargs=generation_kwargs)
    thread.start()
    for new_token in streamer:
        yield new_token
    thread.join()


async def chat_sse(question: str):
    async for new_token in chat(question):
        yield f"data: {new_token}\n\n"
    yield "data: [DONE]\n\n"
    
    
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def main(payload: ChatRequest):

    
    return StreamingResponse(chat_sse(payload.question), media_type="text/event-stream") 
    


# latencies = []

# for prompt in prompts:
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a friendly chatbot who always responds in the style of a pirate",
#         },
#         {"role": "user", "content": prompt},
#     ]
#     prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     start = time.time()
#     outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
#     end = time.time()
    
#     latency  = end - start
#     latencies.append(latency)
#     print(f"Request done in {latency:.2f}s")

#     print(f"Latency: {latency}")
#     print(outputs[0]["generated_text"])
    
# print(f"Average latency: {sum(latencies) / len(latencies):.2f}s")
# print(f"Min: {min(latencies):.2f}s")
# print(f"Max: {max(latencies):.2f}s")
    
    

# with open("latencies.txt", "w") as f:
#     f.write(f"Date: {time.strftime('%Y-%m-%d')}\n")
#     f.write(f"Model: TinyLlama-1.1B\n")
#     f.write(f"Test: 10 sequential requests\n\n")
#     for i, l in enumerate(latencies):
#         f.write(f"Request {i+1}: {l:.2f}s\n")
#     f.write(f"\nAverage latency: {sum(latencies) / len(latencies):.2f}s\n")
#     f.write(f"Min: {min(latencies):.2f}s\n")
#     f.write(f"Max: {max(latencies):.2f}s\n")
    