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
        "plain_title": "Chunked Prefill, FlashInfer and Optimized Concurrency",
        "hypothesis": (
            "Enable chunked prefill with a max-num-batched-tokens of 4096 to prevent prefill requests "
            "from starving decodes and causing huge P99 latency spikes. Set max-num-seqs to 256 to "
            "handle concurrency robustly under load, and use the highly optimized flashinfer "
            "attention backend. Lower gpu-memory-utilization to 0.90 to guarantee safe and successful "
            "CUDA graph capture while avoiding OOM errors under concurrent requests."
        ),
        "category": "vllm_flags",
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.90",
            "--max-num-seqs=256",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens=4096",
            "--attention-backend=flashinfer",
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
