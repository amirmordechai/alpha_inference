# `alpha_inference` — Agent Guide

This repository uses official **Google Cloud AlphaEvolve** (`Google-Cloud-AI/alphaevolve-on-googlecloud`) to autonomously optimize `vLLM` inference flags, CUDA graph ladders, runtime backends, and custom `@triton.jit` GPU kernels for any GPU, model, and OpenAI-compatible JSONL workload trace.

## Onboarding Skill
When asked to set up this repository, check GCP IAM/quotas, create or attach a Vertex AI Endpoint, prepare a trace, or run an optimization experiment, read and follow:
- **[`.agents/skills/alpha-inference/SKILL.md`](.agents/skills/alpha-inference/SKILL.md)**

## Key Files & Boundaries
- **`candidate.py`** — Seed configuration mutated by AlphaEvolve (`# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END`).
- **`optimize.py`** — Evolutionary search runner (`--vertex-endpoint-id`, `--deploy-vertex`, or `--endpoint`; emits native `gcloud_rollout.sh` and supports `--diff`).
- **`validate.py`** — Side-by-side 1:1 real-time unseen trace validator.
- **Cost Safety:** Always run `--dry-run` (`$0` spend) first, and ask the user for explicit approval before provisioning GPUs or running live cloud experiments (`RUN_LIVE_VERTEX_E2E=1`). Always ensure temporary endpoints are torn down (`--auto-teardown` or `gcloud ai endpoints delete`).
