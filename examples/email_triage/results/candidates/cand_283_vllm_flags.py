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
        "plain_title": "Chunked Prefill and Optimized Budgets",
        "hypothesis": (
            "Enabling explicit chunked prefill (--enable-chunked-prefill) with a chunk size of 2048 "
            "prevents long prefill requests from starving concurrent decode steps, directly improving "
            "P99 latency. Removing the experimental V2 model runner eliminates dispatch overhead and "
            "regressions. Tuning block-size to 16 minimizes memory fragmentation, while FLASH_ATTN "
            "and a sequence limit of 256 optimize execution throughput under load."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.90",
            "--language-model-only",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens=2048",
            "--max-num-seqs=256",
            "--block-size=16",
            "--async-scheduling",
            "--attention-backend=FLASH_ATTN",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
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
