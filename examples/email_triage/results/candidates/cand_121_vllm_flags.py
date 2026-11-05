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
        "plain_title": "Async Scheduling & Reduced Max Seqs",
        "hypothesis": (
            "Enable async scheduling to hide CPU scheduling overhead. "
            "Lower max-num-seqs to 128 to reduce queueing delay and avoid starving concurrency, "
            "while maintaining V2 model runner and FlashInfer."
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
            "--disable-uvicorn-access-log",
            "--language-model-only",
            "--enable-prefix-caching",
            "--attention-backend=flashinfer",
            "--async-scheduling",
            "--max-num-seqs=128",
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=1024",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
