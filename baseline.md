## Baseline Benchmark — Sequential (No Batching)

Date: 2026-04-20
Model: TinyLlama-1.1B-Chat
Hardware: NVIDIA RTX 4050 6GB
Max tokens: 200
Prompt: "What is Artificial Intelligence?"

| Users | Successful | Avg TTFT | Avg Latency | Avg Tokens | RPS   |
| ----- | ---------- | -------- | ----------- | ---------- | ----- |
| 1     | 1/1        | 7.480s   | 10.262s     | 150        | 0.097 |
| 5     | 5/5        | 5.812s   | 8.790s      | 139        | 0.329 |
| 10    | 10/10      | 17.882s  | 21.654s     | 135        | 0.260 |

Observation: TTFT increases linearly with queue position —
each user waits for all previous requests to complete.
Sequential throughput stays flat at ~0.1 req/s per request.
Wall time for 10 users: 38.49s

## Naive Batching Benchmark

Model: TinyLlama-1.1B-Chat
Hardware: NVIDIA RTX 4050 6GB
Max tokens: 200
Prompt: "What is Artificial Intelligence?"

## Final Summary

| Users | Success | TTFT  | Latency | Tokens | RPS   |
| ----- | ------- | ----- | ------- | ------ | ----- |
| 1     | 1       | 4.986 | 4.986   | 41.0   | 0.201 |
| 5     | 5       | 3.233 | 3.233   | 54.0   | 1.546 |
| 10    | 10      | 4.581 | 4.581   | 64.4   | 1.442 |

| Users | Baseline RPS | Batched RPS | Improvement |
| ----- | ------------ | ----------- | ----------- |
| 1     | 0.097        | 0.201       | 2.1x        |
| 5     | 0.329        | 1.546       | 4.7x        |
| 10    | 0.260        | 1.442       | 5.5x        |

Wall time for 10 users:
Baseline: 38.49s
Batched: 6.93s → 5.6x faster
