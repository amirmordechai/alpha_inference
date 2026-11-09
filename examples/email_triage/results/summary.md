# Alpha Inference Optimization Report (`COMPLETED`)

- **Updated (UTC):** `2026-11-09 20:14:51 UTC`
- **Official AlphaEvolve Experiment:** `13416136898898249762`
- **Optimized Metric:** `p99_latency` (P99 Latency)
- **Candidates Evaluated:** `302` (`11` record-setting breakthroughs)
- **Estimated Cloud Spend:** `$91.78`

## Ground-Zero Baseline vs. Best Evolved

| Metric | Baseline (`#0`) | Best Evolved | Change |
| :--- | ---: | ---: | :--- |
| **Target (P99 Latency)** | `8943.18 ms` | **`3736.11 ms`** | **`-58.2%`** |
| **P99 Latency** | `8943.18 ms` | `3736.11 ms` | `-58.2%` |
| **Median (P50) Latency** | `6152.53 ms` | `2251.66 ms` | `-63.4%` |
| **Sustained Throughput** | `23.85 req/s` | `38.41 req/s` | `+61.0%` |
| **KL Divergence vs. Baseline** | `0.00000` | `0.00584` | `Exact match: 99.45%` |

## Discovered Breakthroughs

| Eval # | Category | What AlphaEvolve Changed | P99 Latency | Throughput |
| ---: | :--- | :--- | ---: | ---: |
| `#1` | `backends` | Online FP8 quantization and language-model-only | `5431.73 ms` | `32.23 req/s` |
| `#6` | `backends` | Online FP8 quantization and language-model-only | `5406.01 ms` | `32.66 req/s` |
| `#5` | `backends` | FP8 online quant + GPU ngram spec + async LM | `4938.56 ms` | `32.88 req/s` |
| `#49` | `cuda_graphs` | FP8 + N-Gram Spec + Sparse MoE Tile Schedule | `4896.62 ms` | `32.63 req/s` |
| `#128` | `cuda_graphs` | Exact-Divisor N=64 MoE Tiles + 3-Gram Spec | `4887.12 ms` | `33.27 req/s` |
| `#164` | `backends` | FP8 Weights+KV + High-Occupancy MoE Tiles | `4095.39 ms` | `38.98 req/s` |
| `#172` | `backends` | FP8 KV+Weights + SM120 MoE Schedule & Graphs | `3986.40 ms` | `38.34 req/s` |
| `#193` | `cuda_graphs` | 4-Stage FP8 MoE Schedule + 1024 Prefill Chunk | `3971.07 ms` | `38.87 req/s` |
| `#203` | `vllm_flags` | GPU N-Gram SpecDecode + FP8 + SM120 MoE | `3920.93 ms` | `37.61 req/s` |
| `#210` | `vllm_flags` | GPU N-Gram Speculation + Dense 512 CUDA Graphs | `3821.17 ms` | `37.01 req/s` |
| `#238` | `vllm_flags` | 2-Token N-Gram Spec + Sparse M=512 MoE Tiles | `3736.11 ms` | `38.41 req/s` |

## Reproduce & Roll Out to Vertex AI
- Inspect exact diff against baseline: `python optimize.py --diff --out-dir examples/email_triage/results`
- Roll out winning configuration with native `gcloud ai`: `bash examples/email_triage/results/gcloud_rollout.sh`
- Validate on unseen real-time trace: `python validate.py --deploy-vertex --auto-teardown --out-dir examples/email_triage/results`


## Held-Out Unseen Production Validation (1:1 Real-Time Replay)

- **Unseen Trace:** `examples/email_triage/tuesday_peak_30m.jsonl.gz` (`19800` unseen requests over `30.0` real-time minutes)
- **Output Equivalence on Unseen Data:** `mean_kl = 0.00561`, `exact_match = 99.54%`

| Metric | Standalone Baseline | Standalone Best Evolved | Delta |
| :--- | ---: | ---: | :--- |
| **P99 Latency** | `24811.36 ms` | **`21718.67 ms`** | **`-12.5%`** |
| **Median (P50) Latency** | `9111.75 ms` | `3419.41 ms` | `-62.5%` |
| **Sustained Throughput** | `25.16 req/s` | **`40.76 req/s`** | **`+62.0%`** |
