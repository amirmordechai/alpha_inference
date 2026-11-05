---
name: alpha-inference
description: >-
  Onboard a user and run autonomous GenAI inference optimization on any Vertex AI
  GPU Endpoint or GPU VM using official Google Cloud AlphaEvolve. Use this skill
  when the user asks to set up alpha_inference, verify GCP IAM/quotas, create or
  attach a Vertex AI Endpoint, prepare a JSONL workload trace, or run E2E optimization.
---

# `alpha_inference` — Coding Agent Onboarding & E2E Runbook

Follow these steps interactively with the user to set up and run an autonomous inference optimization experiment across any GPU, model, and workload.

## Architecture Overview
- **`candidate.py`** — Single seed file mutated by Google Cloud AlphaEvolve (`# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END`). Controls `vLLM` flags, CUDA graph ladders, runtime env vars, and custom `@triton.jit` GPU kernels.
- **`optimize.py`** — Inspects a live `--vertex-endpoint-id` (or provisions `--deploy-vertex` workers), runs official `AlphaEvolveClient` search with warm `<60s` `vLLM` hot-swaps, enforces the Layer-2 Token-Logprob KL Equivalence Gate, and emits `gcloud_rollout.sh`.
- **`validate.py`** — Replays a disjoint, unseen 1:1 production trace side-by-side on Baseline (`candidate.py`) vs. Best Evolved (`best_candidate.py`).

---

## Step 1: Verify GCP Prerequisites, IAM & Environment

1. Activate `.venv` and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. Verify `.env` (`GCP_PROJECT_ID`, `GE_APP_ID`, `GCP_REGION`) and check prerequisites:
   - **APIs enabled:** `aiplatform.googleapis.com`, `discoveryengine.googleapis.com`, `artifactregistry.googleapis.com`.
   - **Gemini Enterprise:** `GE_APP_ID` points to an AlphaEvolve-enabled Gemini Enterprise engine.
   - **IAM Roles:** `roles/aiplatform.user` and `roles/discoveryengine.admin` (plus `roles/artifactregistry.writer` & `roles/cloudbuild.builds.editor` if mirroring a public Docker Hub `vllm` image).
   - **GPU Quota:** Serving quota available in `GCP_REGION` for the target GPU (`NVIDIA_L4`, `NVIDIA_TESLA_A100`, `NVIDIA_H100_80GB`, `NVIDIA_RTX_PRO_6000`, etc.).

---

## Step 2: Connect an Existing Vertex AI Endpoint (or Create One via `gcloud ai`)

Ask the user whether they already have a live Vertex AI GPU Endpoint ID:

- **Option A — Existing Vertex AI Endpoint (`VERTEX_ENDPOINT_ID`):**
  Pass `--vertex-endpoint-id $VERTEX_ENDPOINT_ID` to `optimize.py`. It automatically inspects the endpoint, inherits its GPU type, machine type, container image, and model, and spins up temporary hot-reload search workers without disturbing live production traffic.

- **Option B — Create a new Vertex AI Endpoint with native `gcloud ai`:**
  ```bash
  VERTEX_ENDPOINT_ID=$(gcloud ai endpoints create \
    --project=$GCP_PROJECT_ID --region=$GCP_REGION \
    --display-name=my-vllm-endpoint \
    --format="value(name)" | awk -F/ '{print $NF}')

  MODEL_ID=$(gcloud ai models upload \
    --project=$GCP_PROJECT_ID --region=$GCP_REGION \
    --display-name=my-vllm-model \
    --container-image-uri=$VLLM_IMAGE_URI \
    --container-command="vllm,serve" \
    --container-args="google/gemma-4-26B-A4B-it,--served-model-name=google/gemma-4-26B-A4B-it,--host=0.0.0.0,--port=8080,--max-model-len=8192,--max-logprobs=16" \
    --container-env-vars="HF_HOME=/dev/shm/hf_cache" \
    --container-shared-memory-size-mb=81920 \
    --container-ports=8080 \
    --container-predict-route=/v1/chat/completions \
    --container-health-route=/health \
    --container-deployment-timeout-seconds=3600 \
    --format="value(model)")

  gcloud ai endpoints deploy-model $VERTEX_ENDPOINT_ID \
    --project=$GCP_PROJECT_ID --region=$GCP_REGION \
    --model="$MODEL_ID" --display-name=my-vllm-model \
    --machine-type=g4-standard-48 \
    --accelerator=type=nvidia-rtx-pro-6000,count=1
  ```

- **Option C — Direct Ephemeral Workers (No Pre-Existing Endpoint Required):**
  Pass `--deploy-vertex --auto-teardown --model <HF_MODEL_ID> --machine-type <MACHINE> --accelerator-type <GPU>` directly to `optimize.py`.

---

## Step 3: Prepare Production-Realistic JSONL Workload Traces

Ensure traces mirror real customer traffic without artificial cache shortcuts:
- Format each JSONL line with `request_id`, `timestamp` (ISO-8601 or float seconds for 1:1 replay), and OpenAI `messages` (`[{"role": "system", ...}, {"role": "user", ...}]`).
- **Critical Anti-Cache-Bias Rule:** Never duplicate/recycle identical user prompts across a trace window (otherwise `vLLM` prefix caching hits artificial 100% cache rates that never happen in real production). Shared system instructions are realistic; user payloads must be unique and disjoint between the **search trace** (`optimize.py`) and **unseen validation trace** (`validate.py`).

---

## Step 4: Run Zero-Cost Dry-Run Checks (`$0` Spend)

```bash
python optimize.py --trace <SEARCH_TRACE.jsonl> --metric p99_latency --dry-run
python validate.py --trace <UNSEEN_TRACE.jsonl> --dry-run
pytest -q
```

---

## Step 5: Confirm Budget & Execute Optimization -> Validation -> Rollout

> **Important Guardrail:** Always state the GPU type, `--workers` count, `--max-programs`, and `--budget-usd` cap to the user and get explicit approval before provisioning live GPUs.

```bash
# 1. Run AlphaEvolve search
python optimize.py \
  --vertex-endpoint-id $VERTEX_ENDPOINT_ID \
  --trace <SEARCH_TRACE.jsonl> \
  --metric p99_latency \
  --max-programs 100 \
  --budget-usd 50 \
  --out-dir results/my_workload

# 2. Validate Before vs. After on unseen 1:1 production traffic
python validate.py \
  --trace <UNSEEN_TRACE.jsonl> \
  --deploy-vertex --auto-teardown \
  --out-dir results/my_workload

# 3. Inspect diff & roll out winning config via generated native `gcloud ai` script
python optimize.py --diff --out-dir results/my_workload
bash results/my_workload/gcloud_rollout.sh
```
