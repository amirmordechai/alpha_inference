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
        "plain_title": "Chunked Prefill with FlashInfer and optimized Batching",
        "hypothesis": (
            "Enabling chunked prefill explicitly with max-num-batched-tokens=4096 and "
            "long-prefill-token-threshold=1024 reduces prefill-decode interference. "
            "Increasing max-num-seqs to 256 prevents queueing delay under high arrival rates. "
            "Reverting block-size to 16 and disabling V2 model runner avoids experimental overheads."
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
            "--language-model-only",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens=4096",
            "--max-num-seqs=256",
            "--long-prefill-token-threshold=1024",
            "--block-size=16",
            "--async-scheduling",
            "--attention-backend=FLASHINFER",
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
