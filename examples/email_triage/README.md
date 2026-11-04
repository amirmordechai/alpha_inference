# Included Reference Benchmark: Enterprise Email Triage on Vertex AI

This folder contains the runnable traces and full results from optimizing **`nvidia/diffusiongemma-26B-A4B-it-NVFP4`** on **NVIDIA RTX PRO 6000 (`g4-standard-48`)** Vertex AI Dedicated Endpoints using official **Google Cloud AlphaEvolve**.

## Trace Files (Disjoint 1:1 Production Slices from a 100k-Email Week)
- **`monday_peak_3m.jsonl`** (`942` emails, `3.0` real-time minutes, `Mon 09:00–09:03`): Exact 1:1 production arrival timestamps used during the AlphaEvolve evolutionary search (`optimize.py`).
- **`equivalence_golden.jsonl`** (`80` stratified emails): Used by the Layer-2 Token-Logprob KL Equivalence Gate before timing any candidate.
- **`tuesday_peak_30m.jsonl`** (`6,148` unseen emails, `30.0` real-time minutes, `Tue 09:00–09:30`): Held-out production peak trace (`0%` overlap with Monday) replayed at 1:1 real-time arrival rates in `validate.py`.

## Articulated Output (`results/`)
- `summary.md` — Executive table comparing Ground-Zero Baseline vs. Best Evolved, every record-setting breakthrough, and held-out 30-minute unseen validation.
- `results.tsv` — Full evaluation ledger (`iteration`, `status`, `category`, `p99_latency_ms`, `p50_latency_ms`, `throughput_rps`, `mean_kl`).
- `best_candidate.py` — Winning evolved `candidate.py` program (diff via `python optimize.py --diff`; roll out to Vertex AI via `bash examples/email_triage/results/gcloud_rollout.sh`).
- `progress.png` — AlphaEvolve optimization trajectory chart.
- `unseen_validation.json` & `unseen_validation.png` — Side-by-side 30-minute unseen production replay telemetry and chart.
