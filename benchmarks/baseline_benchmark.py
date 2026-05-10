import asyncio
from pathlib import Path
import aiohttp
import time

SERVER_URL = "http://localhost:8000"
PROMPT = "What is Artificial Intelligence?"
TIMEOUT = aiohttp.ClientTimeout(total=300)  # 5 min — sequential queue can be slow
RESULTS_PATH = Path(__file__).resolve().parent / "baseline_results.csv"

async def single_request(session, user_id):
    start = time.time()
    first_token_time = None
    token_count = 0

    try:
        async with session.post(
            f"{SERVER_URL}/generate",
            json={"question": PROMPT}
        ) as response:
            async for line in response.content:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: "):
                    token = decoded[6:]
                    if token == "[DONE]":
                        break
                    if first_token_time is None:
                        first_token_time = time.time()
                    token_count += len(token.split())

        ttft = round(first_token_time - start, 3) if first_token_time else None
        total = round(time.time() - start, 3)
        print(f"  User {user_id}: TTFT={ttft}s  Total={total}s  Tokens={token_count}")
        return {"ttft": ttft, "total": total, "tokens": token_count, "success": True}

    except Exception as e:
        total = round(time.time() - start, 3)
        print(f"  User {user_id}: FAILED after {total}s — {e}")
        return {"ttft": None, "total": total, "tokens": 0, "success": False}


async def load_test(n_users):
    print(f"\n{'='*50}")
    print(f"  {n_users} concurrent users")
    print(f"{'='*50}")

    wall_start = time.time()
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        tasks = [single_request(session, i) for i in range(n_users)]
        results = await asyncio.gather(*tasks)
    wall_time = time.time() - wall_start

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        print(f"\n  All {n_users} requests failed.")
        return

    ttfts   = [r["ttft"] for r in successful if r["ttft"] is not None]
    totals  = [r["total"] for r in successful]
    tokens  = [r["tokens"] for r in successful]

    print(f"\n  Results:")
    print(f"  Successful:   {len(successful)}/{n_users}")
    print(f"  Failed:       {len(failed)}/{n_users}")
    print(f"  Avg TTFT:     {round(sum(ttfts)/len(ttfts), 3)}s")
    print(f"  Avg Latency:  {round(sum(totals)/len(totals), 3)}s")
    print(f"  Avg Tokens:   {round(sum(tokens)/len(tokens), 1)}")
    print(f"  Wall time:    {round(wall_time, 3)}s")
    print(f"  Throughput:   {round(len(successful)/wall_time, 3)} req/s")

    return {
        "users": n_users,
        "successful": len(successful),
        "avg_ttft": round(sum(ttfts)/len(ttfts), 3),
        "avg_latency": round(sum(totals)/len(totals), 3),
        "avg_tokens": round(sum(tokens)/len(tokens), 1),
        "throughput_rps": round(len(successful)/wall_time, 3),
    }


async def main():
    print("\nLLM Inference Server — Baseline Benchmark")
    print(f"Prompt: '{PROMPT}'")

    results = []
    for n in [1, 5, 10, 25]:
        result = await load_test(n)
        if result:
            results.append(result)
        # small pause between runs so server queue drains
        await asyncio.sleep(2)

    # save to csv
    import csv
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "users", "successful", "avg_ttft",
            "avg_latency", "avg_tokens", "throughput_rps"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("\n\nFinal Summary")
    print(f"{'Users':<8} {'Success':<10} {'TTFT':<10} {'Latency':<12} {'Tokens':<10} {'RPS'}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['users']:<8} "
            f"{r['successful']:<10} "
            f"{r['avg_ttft']:<10} "
            f"{r['avg_latency']:<12} "
            f"{r['avg_tokens']:<10} "
            f"{r['throughput_rps']}"
        )

    print(f"\nSaved to {RESULTS_PATH}")
    # in baseline_benchmark.py — update the summary note
    # print("\nNote: TTFT increases linearly with user number —")
    # print("User N waits for users 0..N-1 to finish.")
    # print(f"Expected total wall time for 10 users: ~{10 * 8}s")
asyncio.run(main())