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
        "plain_title": "vLLM Prefix Caching & Async Scheduling",
        "hypothesis": (
            "Enabling prefix caching, async scheduling, and reducing max-logprobs "
            "to 5 will minimize prefill latency and scheduling overhead, lowering P99 latency."
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
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
