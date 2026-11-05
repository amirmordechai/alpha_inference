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
        "plain_title": "Optimized Chunked Prefill with Long Prefill Threshold",
        "hypothesis": (
            "Increasing max-num-batched-tokens to 4096 to allow larger, more efficient prefill chunks on the RTX 6000 Ada, "
            "while setting a long-prefill-token-threshold of 2048 to minimize chunking overhead for shorter requests. "
            "This combines the high throughput of larger chunk sizes with the low latency of unchunked short prefills "
            "under concurrent execution."
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
            "--attention-backend=FLASHINFER",
            "--language-model-only",
            "--enable-chunked-prefill",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=4096",
            "--long-prefill-token-threshold=2048",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
