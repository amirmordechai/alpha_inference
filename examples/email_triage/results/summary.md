# Alpha Inference Optimization Report (`COMPLETED`)

- **Updated (UTC):** `2026-11-05 01:58:47 UTC`
- **Official AlphaEvolve Experiment:** `8702438161969431074`
- **Optimized Metric:** `p99_latency` (P99 Latency)
- **Candidates Evaluated:** `301` (`5` record-setting breakthroughs)
- **Estimated Cloud Spend:** `$88.41`

## Ground-Zero Baseline vs. Best Evolved

| Metric | Baseline (`#0`) | Best Evolved | Change |
| :--- | ---: | ---: | :--- |
| **Target (P99 Latency)** | `8898.11 ms` | **`8476.82 ms`** | **`-4.7%`** |
| **P99 Latency** | `8898.11 ms` | `8476.82 ms` | `-4.7%` |
| **Median (P50) Latency** | `6123.15 ms` | `5804.24 ms` | `-5.2%` |
| **Sustained Throughput** | `23.95 req/s` | `21.80 req/s` | `-9.0%` |
| **KL Divergence vs. Baseline** | `0.00000` | `0.00060` | `Exact match: 99.96%` |

## Discovered Breakthroughs

| Eval # | Category | What AlphaEvolve Changed | P99 Latency | Throughput |
| ---: | :--- | :--- | ---: | ---: |
| `#7` | `backends` | High-Throughput Low-Latency FlashInfer & Chunked Prefill | `8608.40 ms` | `24.34 req/s` |
| `#21` | `vllm_flags` | Async Scheduling & V2 Model Runner | `8579.05 ms` | `24.07 req/s` |
| `#59` | `vllm_flags` | Max Batch Tokens 8k + GC & Priority Optimization | `8578.65 ms` | `24.50 req/s` |
| `#106` | `vllm_flags` | Chunked Prefill Max Batch 4k + Global Grad Disable | `8545.01 ms` | `24.16 req/s` |
| `#141` | `vllm_flags` | Chunked Prefill 2k + Block Size 32 | `8476.82 ms` | `21.80 req/s` |

## Reproduce & Roll Out to Vertex AI
- Inspect exact diff against baseline: `python optimize.py --diff --out-dir examples/email_triage/results`
- Roll out winning configuration with native `gcloud ai`: `bash examples/email_triage/results/gcloud_rollout.sh`
- Validate on unseen real-time trace: `python validate.py --deploy-vertex --auto-teardown --out-dir examples/email_triage/results`


## Held-Out Unseen Production Validation (1:1 Real-Time Replay)

- **Unseen Trace:** `examples/email_triage/tuesday_peak_30m.jsonl.gz` (`19800` unseen requests over `30.0` real-time minutes)
- **Output Equivalence on Unseen Data:** `mean_kl = 0.00025`, `exact_match = 99.91%`

| Metric | Standalone Baseline | Standalone Best Evolved | Delta |
| :--- | ---: | ---: | :--- |
| **P99 Latency** | `26166.92 ms` | **`25739.52 ms`** | **`-1.6%`** |
| **Median (P50) Latency** | `8732.56 ms` | `7756.49 ms` | `-11.2%` |
| **Sustained Throughput** | `24.77 req/s` | **`24.70 req/s`** | **`-0.3%`** |
