---
name: alpha-inference
description: >-
  Onboard a user and run autonomous GenAI inference optimization on Vertex AI
  GPU Endpoints or GPU VMs using official Google Cloud AlphaEvolve. Use this
  skill when the user asks to set up alpha_inference, deploy a model endpoint,
  prepare a JSONL workload trace, or launch an optimization and validation run.
---

# `alpha_inference` — Coding Agent Onboarding & Runbook

Follow these steps interactively with the user to set up and run an autonomous inference optimization experiment.

## Architecture Overview (Read First)
- **`candidate.py`** — The single seed file mutated by Google Cloud AlphaEvolve (`# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END`). Controls `vLLM` flags, CUDA graph ladders, runtime env vars, and custom `@triton.jit` GPU kernels.
- **`optimize.py`** — Connects official Google Cloud AlphaEvolve (`AlphaEvolveClient`) to one or more warm GPU endpoints, enforces the Layer-2 Token-Logprob KL Equivalence Gate, and replays a 1:1 timestamped trace.
- **`validate.py`** — Replays a disjoint, unseen 1:1 production trace side-by-side on Baseline (`candidate.py`) vs. Best Evolved (`best_candidate.py`).
- **`deploy.py`** — Deploys any model or evolved `best_candidate.py` to a Vertex AI Dedicated GPU Endpoint, prints `--diff`, or tears endpoints down (`--delete`).

---

## Step 1: Verify Python Environment & Google Cloud Credentials

1. Create or activate `.venv` and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. Check `.env` (copy from `.env.example` if missing) and verify with the user:
   - `GCP_PROJECT_ID` — Google Cloud project with Vertex AI enabled.
   - `GE_APP_ID` — AlphaEvolve-enabled Gemini Enterprise engine ID.
   - `GCP_REGION` — GPU region (default `us-central1`).
   - Run `gcloud auth application-default login` if Application Default Credentials are not configured.

---

## Step 2: Connect or Deploy a Live Vertex AI GPU Endpoint

Ask the user which setup applies:

- **Option A — User already has a live Vertex AI Endpoint ID (`<ENDPOINT_ID>`):**
  - If it was deployed with `--hot-reload`, pass `--vertex-endpoint-id <ENDPOINT_ID>` directly to `optimize.py`.
  - If it is a standard Vertex Endpoint without hot-reload enabled yet, either enable hot-reload on it:
    ```bash
    python deploy.py --endpoint-id <ENDPOINT_ID> --hot-reload --deploy --out-json endpoint.json
    ```
    or pass `--vertex-endpoint-id <ENDPOINT_ID> --deploy-vertex --auto-teardown` to `optimize.py` to clone its model onto temporary search workers without touching live traffic.

- **Option B — User needs a live Vertex AI GPU Endpoint first (1 command):**
  ```bash
  python deploy.py --model google/gemma-4-26B-A4B-it --hot-reload --deploy --out-json endpoint.json
  ```

- **Option C — Fully ephemeral multi-worker search:**
  Pass `--deploy-vertex --workers 4 --auto-teardown` to `optimize.py` so endpoints are provisioned only for the run and deleted automatically on exit.

---

## Step 3: Prepare or Select the Workload Trace

Check whether the user wants to run the included benchmark (`examples/email_triage/`) or their own workload:
- Custom traces are standard JSONL files where each line has `request_id`, `timestamp` (ISO-8601 or float seconds for 1:1 replay), and OpenAI `messages` (see [`examples/custom_openai_chat/sample_trace.jsonl`](../../../examples/custom_openai_chat/sample_trace.jsonl)).
- Split custom logs into two **disjoint** time windows:
  1. A short **search peak window** (e.g., 3 minutes of peak traffic) for `--trace` in `optimize.py`.
  2. A **held-out unseen window** (e.g., 15–30 minutes from another day) for `--trace` in `validate.py`.

---

## Step 4: Run Zero-Cost Dry-Run Checks (`$0` Spend)

Always verify configuration, trace parsing, and AST validation before provisioning GPUs:

```bash
python optimize.py --trace <SEARCH_TRACE.jsonl> --metric p99_latency --dry-run
python validate.py --trace <UNSEEN_TRACE.jsonl> --dry-run
pytest -q
```

---

## Step 5: Confirm Budget & Launch Optimization

> **Important Guardrail:** Always state the target GPU count, `--max-programs`, and `--budget-usd` cap to the user and get explicit confirmation before launching live GPU compute.

1. **Run AlphaEvolve optimization:**
   ```bash
   # Against a live Vertex AI Endpoint:
   python optimize.py \
     --vertex-endpoint-id <ENDPOINT_ID> \
     --trace <SEARCH_TRACE.jsonl> \
     --metric p99_latency \
     --max-programs 100 \
     --budget-usd 50 \
     --out-dir results/my_workload
   ```
   *(Or replace `--vertex-endpoint-id <ENDPOINT_ID>` with `--endpoint-json endpoint.json` or `--deploy-vertex --workers 4 --auto-teardown`.)*

2. **Validate Before vs. After on unseen traffic:**
   ```bash
   python validate.py \
     --trace <UNSEEN_TRACE.jsonl> \
     --deploy-vertex --auto-teardown \
     --out-dir results/my_workload
   ```

3. **Review discoveries & roll out to production:**
   ```bash
   python deploy.py --config results/my_workload/best_candidate.py --diff
   python deploy.py --config results/my_workload/best_candidate.py --endpoint-id <ENDPOINT_ID> --deploy
   ```
