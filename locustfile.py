from locust import HttpUser, task, between, events, constant
import csv
import asyncio
import time

ttft_list = []
latency_list = []


# instial benchmarking code for locust load testing
class MyUser(HttpUser):
    wait_time = constant(0)# Simulate user wait time between requests
    host = "http://localhost:8000"  # Base URL of the FastAPI server
    
    @task
    def worker(self):
        start = time.time()
        with self.client.post(
            "/generate",
            json={"question": "What is Artificial Intelligence?"},
            stream=True,
            catch_response=True
        ) as response:

            first_token_time = None

            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        token = decoded[6:]
                        if token == "[DONE]":
                            break
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft_list.append(first_token_time - start)

            total_time = time.time() - start
            latency_list.append(total_time)
            print(f"Total latency: {total_time:.2f}s")
            
            
@events.test_start.add_listener
def reset_lists(environment, **kwargs):
    global ttft_list, latency_list
    ttft_list.clear()
    latency_list.clear()
    print("Lists reset for new test run")
            
@events.quitting.add_listener
def write_results(environment, **kwargs):
    
    user_count = environment.runner.target_user_count 
    if not ttft_list:
        return

    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_latency = sum(latency_list) / len(latency_list)
    rps = environment.stats.total.current_rps

    with open("baseline_results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([
            "users", "avg_ttft", "avg_latency", "rps"
        ])
        writer.writerow([
            user_count,
            round(avg_ttft, 3),
            round(avg_latency, 3),
            round(rps, 3)
        ])

    print(f"\n=== BASELINE BENCHMARK RESULTS ===")
    print(f"Users:       {user_count}")
    print(f"Avg TTFT:    {avg_ttft:.2f}s")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"RPS:         {rps:.2f}")
    print(f"Total reqs:  {len(latency_list)}")  