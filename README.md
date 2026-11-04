# `alpha_inference` — GenAI Inference Optimization with Google Cloud AlphaEvolve

`alpha_inference` connects **official Google Cloud AlphaEvolve** (`Google-Cloud-AI/alphaevolve-on-googlecloud`) to a **Vertex AI GPU Endpoint** (or **GPU VM**) to automatically discover faster `vLLM` serving flags, CUDA graph capture ladders, runtime backends, and custom `@triton.jit` GPU kernels for your exact workload trace.

Every candidate is verified on live GPU hardware behind a **Layer-2 Token-Logprob KL Equivalence Gate** so latency and throughput gains never degrade model quality.

---

## Running with a Coding Agent

Spin up your coding agent (Antigravity, Gemini CLI, Claude Code, Codex, or Cursor) in this repo and prompt:

```text
Hi, have a look at .agents/skills/alpha-inference/SKILL.md and set me up to optimize inference for my model and workload!
```

The [`.agents/skills/alpha-inference/SKILL.md`](.agents/skills/alpha-inference/SKILL.md) runbook walks your agent through environment setup, deploying or attaching a live Vertex AI GPU Endpoint, formatting your JSONL trace, verifying with `$0` `--dry-run` checks, and launching optimization safely.

---

## How It Works

1. **Seed (`candidate.py`):** Defines your starting `vLLM` configuration inside `# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END`.
2. **Evolve (`optimize.py`):** Official Google Cloud AlphaEvolve mutates `candidate.py`, hot-swaps `vLLM` in place on warm GPU endpoints (without cold cloud VM redeployments), checks token-logprob KL equivalence, and replays your production trace at **1:1 real-time arrival timestamps**.
3. **Validate & Deploy (`validate.py` / `deploy.py`):** Replays an **unseen** production trace side-by-side (Baseline vs. Best Evolved) and rolls out the winning configuration to your Vertex AI Endpoint.

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

### 2. Connect a Live Vertex AI Endpoint (or Deploy One in 1 Command)

If you don't have a live Vertex AI GPU Endpoint yet, deploy any HuggingFace model with warm hot-reload enabled in one command:
```bash
python deploy.py \
  --model google/gemma-4-26B-A4B-it \
  --hot-reload --deploy \
  --out-json endpoint.json
```

*(Already have a live Vertex AI Endpoint ID? Enable hot-reload on it with `python deploy.py --endpoint-id $VERTEX_ENDPOINT_ID --hot-reload --deploy`, or let `optimize.py` spin up temporary parallel workers automatically with `--deploy-vertex --workers 4 --auto-teardown`.)*

### 3. Optimize, Validate & Roll Out

```bash
# 1. Optimize directly against your live Vertex AI Endpoint:
python optimize.py \
  --vertex-endpoint-id $VERTEX_ENDPOINT_ID \
  --trace examples/email_triage/monday_peak_3m.jsonl \
  --metric p99_latency \
  --max-programs 100 \
  --budget-usd 50

# 2. Replay a held-out unseen production window side-by-side (Before vs. After):
python validate.py \
  --trace examples/email_triage/tuesday_peak_30m.jsonl \
  --deploy-vertex --auto-teardown

# 3. Inspect the discovered diff and roll out the winner to your Vertex Endpoint:
python deploy.py --diff
python deploy.py \
  --config examples/email_triage/results/best_candidate.py \
  --endpoint-id $VERTEX_ENDPOINT_ID --deploy
```

| Key Flag | Options / Purpose |
| :--- | :--- |
| `--vertex-endpoint-id` | Attach directly to one or more live Vertex AI GPU Endpoint IDs (auto-resolves Dedicated DNS & hot-reload token). |
| `--endpoint-json` | Attach using the JSON metadata file written by `deploy.py --out-json`. |
| `--deploy-vertex` | Automatically provisions warm `NVIDIA RTX PRO 6000` (`g4-standard-48`) Vertex AI Dedicated Endpoints (`--auto-teardown` deletes them on exit). |
| `--endpoint` | Attach to existing GPU VM(s) running `python -m alpha_inference.reload_proxy`. |
| `--trace` | Path to your JSONL request trace (`messages`/`prompt` chat or structured task). |
| `--metric` | `p99_latency` *(minimize)*, `latency` *(minimize P50)*, `throughput` *(maximize req/s)*, `ttft` *(minimize P99 Time-To-First-Token)*. |
| `--budget-usd` | Hard USD cloud-spend circuit breaker (`BudgetedAlphaEvolveExperiment`). |

---

## Articulated Output Format

Every run populates `--out-dir` (see [`examples/email_triage/results/`](examples/email_triage/results/)) with a standardized set of artifacts:

- **`summary.md`** — Human-readable report with Baseline vs. Best metrics, discovered breakthroughs, and unseen validation table.
- **`results.tsv`** — Tab-separated ledger of every evaluated candidate, status (`keep`, `discard`, `crash`, `invalid`), latency, throughput, and KL divergence.
- **`best_candidate.py`** — Winning evolved Python configuration (deployable directly with `python deploy.py --config <path> --deploy`).
- **`progress.png`** — Optimization trajectory plot across evaluations.
- **`unseen_validation.json` & `unseen_validation.png`** — Held-out 1:1 production replay results.

---

## Repository Structure

```text
alpha_inference/
├── AGENTS.md                   # Coding agent overview & guardrails
├── .agents/skills/             # Agent onboarding skill (`alpha-inference/SKILL.md`)
├── optimize.py                 # Official Google Cloud AlphaEvolve optimization CLI
├── validate.py                 # Side-by-side 1:1 real-time unseen trace validator
├── candidate.py                # Seed program (# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END)
├── deploy.py                   # Vertex AI Endpoint deployer (`--model`, `--config`, `--diff`, `--delete`)
├── alpha_inference/            # Core library
│   ├── adapters.py             # OpenAI chat + SSE TTFT + structured diffusion adapters
│   ├── equivalence.py          # Layer-2 token-logprob KL divergence anti-cheating gate
│   ├── evaluator.py            # 1:1 real-time timestamp replay & throughput probe
│   ├── reload_proxy.py         # Warm GPU hot-reload supervisor & Kineto profiler
│   ├── reporting.py            # AST safety validator, Vertex inspector, plots & summary
│   └── problem.md              # Optimization contract & search-space guide for AlphaEvolve
├── examples/
│   ├── email_triage/           # Reference benchmark traces + full articulated results
│   └── custom_openai_chat/     # Template trace & guide for standard OpenAI chat models
└── tests/
    └── test_alpha_inference.py # Offline unit & integration test suite ($0 cloud spend)
```
