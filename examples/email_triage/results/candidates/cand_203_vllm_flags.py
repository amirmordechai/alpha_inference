"""Seed inference configuration evolved by Google Cloud AlphaEvolve (`candidate.py`).

AlphaEvolve mutates the function inside the evolve block below to discover faster
vLLM scheduler flags, CUDA graph capture ladders, runtime backends, and custom
Triton GPU kernels tailored to your workload trace and GPU hardware.
"""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
# IMPORTANT: Keep the function name `get_candidate_config` and return dict keys unchanged.
def get_candidate_config() -> dict[str, Any]:
    """Return the vLLM serving configuration and optional custom Triton GPU kernel."""
    return {
        "plain_title": "Crazy Language-Only Lobotomy & Triton Micro-Chunking",
        "hypothesis": (
            "Crazy Idea (capital C): Completely bypass multimodal overhead by forcing "
            "--language-model-only. This massive VRAM/token-budget saving allows us to "
            "aggressively slice prefills with --max-num-batched-tokens=4096 without ValueError. "
            "Switching to TRITON_ATTN avoids head_dim>256 crashes from flashinfer. We drop V2 "
            "runner, use async scheduling, and boost OMP_NUM_THREADS to stabilize engine startup."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.95",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--language-model-only",
            "--max-num-seqs=128",
            "--max-num-batched-tokens=4096",
            "--async-scheduling",
            "--attention-backend=TRITON_ATTN",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
