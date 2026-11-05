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
        "plain_title": "Optimized Concurrency & Large Batch Thresholds",
        "hypothesis": (
            "The high TTFT P99 suggests excessive prefill chunking overhead or queueing is dominating latency. "
            "By increasing max-num-batched-tokens to 16384, introducing long-prefill-token-threshold, and reducing max-num-seqs to 128, "
            "we minimize TTFT while maintaining the throughput benefits of chunked prefill."
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
            "--max-num-batched-tokens=16384",
            "--max-num-seqs=128",
            "--long-prefill-token-threshold=2048",
            "--enable-prefix-caching",
            "--async-scheduling",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "TOKENIZERS_PARALLELISM": "true",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
