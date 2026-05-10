FROM python:3.11-slim

# install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY main.py .

# download model at build time so container starts fast
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]