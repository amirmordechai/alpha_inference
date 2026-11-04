# Alpha Inference Optimization Report (`COMPLETED`)

- **Updated (UTC):** `2026-11-03 22:06:15 UTC`
- **Official AlphaEvolve Experiment:** `10256809673523924228`
- **Optimized Metric:** `p99_latency` (P99 Latency)
- **Candidates Evaluated:** `242` (`2` record-setting breakthroughs)
- **Estimated Cloud Spend:** `$65.01`

## Ground-Zero Baseline vs. Best Evolved

| Metric | Baseline (`#0`) | Best Evolved | Change |
| :--- | ---: | ---: | :--- |
| **Target (P99 Latency)** | `992.13 ms` | **`952.11 ms`** | **`-4.0%`** |
| **P99 Latency** | `992.13 ms` | `952.11 ms` | `-4.0%` |
| **Median (P50) Latency** | `548.31 ms` | `511.86 ms` | `-6.6%` |
| **Sustained Throughput** | `46.49 req/s` | `46.92 req/s` | `+0.9%` |
| **KL Divergence vs. Baseline** | `0.00000` | `0.07183` | `Exact match: 94.17%` |

## Discovered Breakthroughs

| Eval # | Category | What AlphaEvolve Changed | P99 Latency | Throughput |
| ---: | :--- | :--- | ---: | ---: |
| `#1` | `vllm_flags` | Optimized Micro-Batching & Minimal Logprobs | `976.57 ms` | `44.22 req/s` |
| `#136` | `cuda_graphs` | Optimized CUDA Graph Ladder & Moderate Micro-Batching | `952.11 ms` | `46.92 req/s` |

## Reproduce & Roll Out to Vertex AI
- Inspect exact diff against baseline: `python optimize.py --diff`
- Roll out winning configuration with native `gcloud ai`: `bash examples/email_triage/results/gcloud_rollout.sh`
- Validate on unseen real-time trace: `python validate.py --deploy-vertex --auto-teardown`


## Held-Out Unseen Production Validation (1:1 Real-Time Replay)

- **Unseen Trace:** `examples/email_triage/tuesday_peak_30m.jsonl` (`6148` unseen requests over `30.0` real-time minutes)
- **Output Equivalence on Unseen Data:** `mean_kl = 0.06673`, `exact_match = 94.00%`

| Metric | Standalone Baseline | Standalone Best Evolved | Delta |
| :--- | ---: | ---: | :--- |
| **P99 Latency** | `1146.89 ms` | **`1115.43 ms`** | **`-2.7%`** |
| **Median (P50) Latency** | `338.10 ms` | `343.50 ms` | `+1.6%` |
| **Sustained Throughput** | `45.43 req/s` | **`44.81 req/s`** | **`-1.4%`** |
