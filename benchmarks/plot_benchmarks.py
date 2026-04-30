from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

users = [1, 5, 10]
baseline_rps = [0.097, 0.329, 0.260]
batched_rps  = [0.201, 1.546, 1.442]
baseline_wall = [10.26, 15.22, 38.49]
batched_wall  = [4.99,  3.24,  6.93]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(users))
width = 0.35

# RPS chart
ax1.bar(x - width/2, baseline_rps, width, label='Baseline', color='#e74c3c', alpha=0.8)
ax1.bar(x + width/2, batched_rps,  width, label='Batched',  color='#2ecc71', alpha=0.8)
ax1.set_xlabel('Concurrent Users')
ax1.set_ylabel('Requests per Second')
ax1.set_title('Throughput: Baseline vs Batched')
ax1.set_xticks(x)
ax1.set_xticklabels(users)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for i, (b, a) in enumerate(zip(baseline_rps, batched_rps)):
    ax1.annotate(f'{a/b:.1f}x', xy=(x[i] + width/2, a),
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

# Wall time chart
ax2.bar(x - width/2, baseline_wall, width, label='Baseline', color='#e74c3c', alpha=0.8)
ax2.bar(x + width/2, batched_wall,  width, label='Batched',  color='#2ecc71', alpha=0.8)
ax2.set_xlabel('Concurrent Users')
ax2.set_ylabel('Wall Time (seconds)')
ax2.set_title('Wall Time: Baseline vs Batched')
ax2.set_xticks(x)
ax2.set_xticklabels(users)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('TinyLlama-1.1B — RTX 4050 6GB', fontsize=13, fontweight='bold')
plt.tight_layout()
output_path = Path(__file__).resolve().parents[1] / "benchmark_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved {output_path}")
plt.show()