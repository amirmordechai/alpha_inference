# `alpha_inference` — Agent Guide

This repository uses official **Google Cloud AlphaEvolve** (`Google-Cloud-AI/alphaevolve-on-googlecloud`) to autonomously optimize `vLLM` serving flags, CUDA graph ladders, runtime backends, and custom `@triton.jit` GPU kernels for a user's live workload trace.

## Onboarding Skill
When asked to set up this repository, deploy a Vertex AI Endpoint, prepare a trace, or run an optimization experiment, read and follow:
- **[`.agents/skills/alpha-inference/SKILL.md`](.agents/skills/alpha-inference/SKILL.md)**

## Key Files & Boundaries
- **`candidate.py`** — Seed configuration mutated by AlphaEvolve (`# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END`).
- **`optimize.py`** — Evolutionary search runner (`--vertex-endpoint-id`, `--endpoint-json`, `--endpoint`, or `--deploy-vertex`).
- **`validate.py`** — Side-by-side 1:1 real-time unseen trace validator.
- **`deploy.py`** — One-command Vertex AI Dedicated GPU Endpoint deployer (`--model`, `--config`, `--hot-reload`, `--diff`, `--delete`).
- **Cost Safety:** Always run `--dry-run` (`$0` spend) first, and ask the user for explicit approval before provisioning GPUs or running cloud experiments. Always ensure temporary endpoints are torn down (`--auto-teardown` or `python deploy.py --delete --endpoint-id <ID>`).
