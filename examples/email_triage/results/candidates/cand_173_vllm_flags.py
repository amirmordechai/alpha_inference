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
        "plain_title": "Maximized Throughput via Deep Batching & Pre-Caching",
        "hypothesis": (
            "Crazy idea: Massive batching over-provisioning to completely eliminate queueing delays. "
            "By scaling max-num-seqs to 256 and max-num-batched-tokens to 16384, we maximize GPU "
            "utilization and throughput. A higher gpu-memory-utilization of 0.95 ensures ample KV "
            "cache capacity, combined with block-size=16 for tight memory packing."
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
            "--attention-backend=flashinfer",
            "--language-model-only",
            "--max-num-seqs=256",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens=16384",
            "--long-prefill-token-threshold=8192",
            "--block-size=16",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "2",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
