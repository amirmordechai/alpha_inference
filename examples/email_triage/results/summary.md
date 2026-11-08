# Alpha Inference Optimization Report (`COMPLETED`)

- **Updated (UTC):** `2026-11-08 14:10:55 UTC`
- **Official AlphaEvolve Experiment:** `4156422443182074563`
- **Optimized Metric:** `p99_latency` (P99 Latency)
- **Candidates Evaluated:** `543` (`8` record-setting breakthroughs)
- **Estimated Cloud Spend:** `$136.24`

## Ground-Zero Baseline vs. Best Evolved

| Metric | Baseline (`#0`) | Best Evolved | Change |
| :--- | ---: | ---: | :--- |
| **Target (P99 Latency)** | `8898.11 ms` | **`3845.76 ms`** | **`-56.8%`** |
| **P99 Latency** | `8898.11 ms` | `3845.76 ms` | `-56.8%` |
| **Median (P50) Latency** | `6123.15 ms` | `2290.09 ms` | `-62.6%` |
| **Sustained Throughput** | `23.95 req/s` | `36.85 req/s` | `+53.9%` |
| **KL Divergence vs. Baseline** | `0.00000` | `0.00517` | `Exact match: 99.58%` |

## Discovered Breakthroughs

| Eval # | Category | What AlphaEvolve Changed | P99 Latency | Throughput |
| ---: | :--- | :--- | ---: | ---: |
| `#7` | `backends` | High-Throughput Low-Latency FlashInfer & Chunked Prefill | `8608.40 ms` | `24.34 req/s` |
| `#21` | `vllm_flags` | Async Scheduling & V2 Model Runner | `8579.05 ms` | `24.07 req/s` |
| `#59` | `vllm_flags` | Max Batch Tokens 8k + GC & Priority Optimization | `8578.65 ms` | `24.50 req/s` |
| `#106` | `vllm_flags` | Chunked Prefill Max Batch 4k + Global Grad Disable | `8545.01 ms` | `24.16 req/s` |
| `#141` | `vllm_flags` | Chunked Prefill 2k + Block Size 32 | `8476.82 ms` | `21.80 req/s` |
| `#301` | `backends` | FP8 Online Quantization & FP8 KV Cache | `4034.64 ms` | `37.97 req/s` |
| `#315` | `backends` | FP8 + Triton MoE Tuning + N-gram Speculative Decoding | `3913.16 ms` | `35.90 req/s` |
| `#435` | `backends` | Optimized Triton MoE Tile Sizes + Enhanced Speculative D | `3845.76 ms` | `36.85 req/s` |

## Reproduce & Roll Out to Vertex AI
- Inspect exact diff against baseline: `python optimize.py --diff --out-dir examples/email_triage/results`
- Roll out winning configuration with native `gcloud ai`: `bash examples/email_triage/results/gcloud_rollout.sh`
- Validate on unseen real-time trace: `python validate.py --deploy-vertex --auto-teardown --out-dir examples/email_triage/results`


## Held-Out Unseen Production Validation (1:1 Real-Time Replay)

- **Unseen Trace:** `examples/email_triage/tuesday_peak_30m.jsonl.gz` (`19800` unseen requests over `30.0` real-time minutes)
- **Output Equivalence on Unseen Data:** `mean_kl = 0.00415`, `exact_match = 99.50%`

| Metric | Standalone Baseline | Standalone Best Evolved | Delta |
| :--- | ---: | ---: | :--- |
| **P99 Latency** | `24603.03 ms` | **`22595.40 ms`** | **`-8.2%`** |
| **Median (P50) Latency** | `9188.79 ms` | `3524.22 ms` | `-61.6%` |
| **Sustained Throughput** | `25.16 req/s` | **`39.55 req/s`** | **`+57.2%`** |
