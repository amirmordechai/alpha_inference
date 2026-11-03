# `alpha_inference` — GenAI Serving Optimization with Google Cloud AlphaEvolve

`alpha_inference` connects **official Google Cloud AlphaEvolve** (`Google-Cloud-AI/alphaevolve-on-googlecloud`) to a **Vertex AI GPU Endpoint** (or **GPU VM**) to automatically discover faster `vLLM` serving flags, CUDA graph capture ladders, runtime backends, and custom `@triton.jit` GPU kernels for your exact workload trace.

Every candidate is verified on live GPU hardware behind a **Layer-2 Token-Logprob KL Equivalence Gate** so latency and throughput gains never degrade model quality.

---

## How It Works

1. **Seed (`candidate.py`):** Defines your starting `vLLM` configuration inside `# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END`.
2. **Evolve (`optimize.py`):** Official Google Cloud AlphaEvolve mutates `candidate.py`, hot-swaps warm GPU endpoints in **~55 seconds** (instead of 35-minute cold cloud redeploys), checks token-logprob KL equivalence, and replays your production trace at **1:1 real-time arrival timestamps**.
3. **Validate (`validate.py`):** Deploys standalone Baseline (`candidate.py`) and Best (`best_candidate.py`) endpoints side-by-side via `deploy.py` and replays an **unseen** production trace to confirm real-world gains.

---

## Quickstart

### 1. Install & Configure Google Cloud Auth
Requires Python `3.10+`, a Google Cloud project with **Vertex AI** enabled, and an **AlphaEvolve-enabled Gemini Enterprise engine** (`GE_APP_ID`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your GCP_PROJECT_ID and GE_APP_ID
gcloud auth application-default login
```

Verify your trace and configuration locally with **`$0` cloud spend**:
```bash
python optimize.py --dry-run
python validate.py --dry-run
```

### 2. Run Optimization (`optimize.py`)
Plug in your **trace**, **GPU endpoint** (`--deploy-vertex` or `--endpoint`), **metric**, and **research budget**:

```bash
python optimize.py \
  --trace examples/email_triage/monday_peak_3m.jsonl \
  --equiv-trace examples/email_triage/equivalence_golden.jsonl \
  --deploy-vertex --workers 4 --auto-teardown \
  --metric p99_latency \
  --max-programs 200 \
  --budget-usd 160 \
  --out-dir examples/email_triage/results
```

| Key Flag | Options / Purpose |
| :--- | :--- |
| `--trace` | Path to your JSONL request trace (`messages`/`prompt` chat or structured task). |
| `--metric` | `p99_latency` *(minimize)*, `latency` *(minimize P50)*, `throughput` *(maximize req/s)*, `ttft` *(minimize P99 Time-To-First-Token)*. |
| `--deploy-vertex` | Automatically provisions warm `NVIDIA RTX PRO 6000` (`g4-standard-48`) Vertex AI Dedicated Endpoints (`--auto-teardown` deletes them on exit). |
| `--endpoint` | Attach to existing GPU VM(s) or Vertex Endpoint(s) running `python -m alpha_inference.reload_proxy`. |
| `--budget-usd` | Hard USD cloud-spend circuit breaker (`BudgetedAlphaEvolveExperiment`). |

### 3. Validate on Unseen Traffic & Deploy (`validate.py` / `deploy.py`)
```bash
# Inspect every flag, env var, and Triton kernel AlphaEvolve discovered:
python deploy.py --diff

# Replay held-out 30-minute unseen production peak side-by-side:
python validate.py \
  --trace examples/email_triage/tuesday_peak_30m.jsonl \
  --deploy-vertex --auto-teardown \
  --out-dir examples/email_triage/results

# Deploy the winning configuration to production:
python deploy.py --config examples/email_triage/results/best_candidate.py --deploy
```

---

## Articulated Output Format

Every run populates `--out-dir` (see [`examples/email_triage/results/`](examples/email_triage/results/)) with a clean, standardized set of artifacts:

- **`summary.md`** — Human-readable report with Baseline vs. Best metrics, discovered breakthroughs, and unseen validation table.
- **`results.tsv`** — Tab-separated ledger of every evaluated candidate, status (`keep`, `discard`, `crash`, `invalid`), latency, throughput, and KL divergence.
- **`best_candidate.py`** — Winning evolved Python configuration (deployable directly with `python deploy.py --config <path> --deploy`).
- **`progress.png`** — Optimization trajectory plot across evaluations.
- **`unseen_validation.json` & `unseen_validation.png`** — Held-out 1:1 production replay results.

---

## Repository Structure

```text
alpha_inference/
├── optimize.py                 # Official Google Cloud AlphaEvolve optimization CLI
├── validate.py                 # Side-by-side 1:1 real-time unseen trace validator
├── candidate.py                # Seed program (# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END)
├── deploy.py                   # Standalone Vertex AI Endpoint deployer (`--config`, `--diff`, `--delete`)
├── alpha_inference/            # Core library
│   ├── adapters.py             # OpenAI chat + SSE TTFT + structured diffusion adapters
│   ├── equivalence.py          # Layer-2 token-logprob KL divergence anti-cheating gate
│   ├── evaluator.py            # 1:1 real-time timestamp replay & throughput probe
│   ├── reload_proxy.py         # ~55s warm GPU hot-reload supervisor & Kineto profiler
│   ├── reporting.py            # AST safety validator, plots, and summary generator
│   └── problem.md              # Optimization contract & search-space guide for AlphaEvolve
├── examples/
│   ├── email_triage/           # Reference benchmark traces + full articulated results
│   └── custom_openai_chat/     # Template trace & guide for standard OpenAI chat models
└── tests/
    └── test_alpha_inference.py # Offline unit & integration test suite ($0 cloud spend)
```
