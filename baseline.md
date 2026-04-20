## Baseline Benchmark — Sequential (No Batching)

Date: 2026-04-20
Model: TinyLlama-1.1B-Chat
Hardware: NVIDIA RTX 4050 6GB
Max tokens: 200
Prompt: "What is Artificial Intelligence?"

| Users | Successful | Avg TTFT | Avg Latency | Avg Tokens | RPS   |
|-------|------------|----------|-------------|------------|-------|
| 1     | 1/1        | 7.480s   | 10.262s     | 150        | 0.097 |
| 5     | 5/5        | 5.812s   | 8.790s      | 139        | 0.329 |
| 10    | 10/10      | 17.882s  | 21.654s     | 135        | 0.260 |

Observation: TTFT increases linearly with queue position —
each user waits for all previous requests to complete.
Sequential throughput stays flat at ~0.1 req/s per request.
Wall time for 10 users: 38.49s