# Included Reference Benchmark: Enterprise Email Triage on Vertex AI

This folder contains the runnable traces and full results from optimizing **`google/gemma-4-26B-A4B-it`** (`26B` total / `4B` active 128-expert MoE) on **NVIDIA RTX PRO 6000 (`g4-standard-48`)** Vertex AI Dedicated Endpoints using official **Google Cloud AlphaEvolve**.

## Trace Files (Disjoint, Zero-Recycled 1:1 Production Slices from a `98,267`-Email Week)
Every request across all files uses a **100% unique user email sampled without replacement** (`0%` recycled prompts within or across traces, so `vLLM` prefix caching only caches the genuine shared system rubric just like real production traffic):
- **`monday_peak_3m.jsonl`** (`3,956` unique emails, `3.0` real-time minutes, `Mon 09:00–09:03`, `~92%` sustained GPU load with `>115%` bursts): Exact 1:1 production arrival timestamps used during the AlphaEvolve evolutionary search (`optimize.py`).
- **`equivalence_golden.jsonl`** (`80` stratified unique emails): Used by the Layer-2 Token-Logprob KL Equivalence Gate before timing any candidate.
- **`tuesday_peak_30m.jsonl.gz`** (`19,800` unseen unique emails, `30.0` real-time minutes, `Tue 09:00–09:30`): Held-out production peak trace (`0%` overlap with Monday or Golden Equivalence) replayed at 1:1 real-time arrival rates in `validate.py`.

## Articulated Output (`results/`)
- `summary.md` — Executive table comparing Ground-Zero Baseline vs. Best Evolved, every record-setting breakthrough, and held-out 30-minute unseen validation.
- `results.tsv` — Full evaluation ledger (`iteration`, `status`, `category`, `p99_latency_ms`, `p50_latency_ms`, `throughput_rps`, `mean_kl`).
- `best_candidate.py` — Winning evolved `candidate.py` program (diff via `python optimize.py --diff`; roll out to Vertex AI via `bash examples/email_triage/results/gcloud_rollout.sh`).
- `progress.png` — AlphaEvolve optimization trajectory chart.
- `unseen_validation.json` & `unseen_validation.png` — Side-by-side 30-minute unseen production replay telemetry and chart.
