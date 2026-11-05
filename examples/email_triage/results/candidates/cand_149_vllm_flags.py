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
        "plain_title": "Balanced Chunked Prefill and V2 Runner",
        "hypothesis": (
            "The previous extreme micro-chunking strategy collapsed throughput and spiked latency. "
            "We restore balance by increasing max-num-batched-tokens to 8192 and dropping the "
            "restrictive prefill token threshold. We lower max-num-seqs to 128 to prevent "
            "concurrency overload, increase gpu-memory-utilization to 0.95 for maximum KV cache, "
            "and rely on the V2 Model Runner for efficient host-device execution."
        ),
        "category": "vllm_flags",
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.95",
            "--max-num-seqs=128",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--max-num-batched-tokens=8192",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "OMP_NUM_THREADS": "8",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
