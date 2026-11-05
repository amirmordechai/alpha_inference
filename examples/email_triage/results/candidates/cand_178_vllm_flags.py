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
        "plain_title": "Optimized Chunked Prefill & Extended CUDA Graph Capture",
        "hypothesis": (
            "Optimize latency by reducing max-num-batched-tokens to 2048 to force effective "
            "chunked prefill chunking, avoiding large prefill stalls. Increase block-size to 32 "
            "for better FlashInfer alignment, and use max-seq-len-to-capture=4096 to expand "
            "CUDA graph coverage for longer sequences without causing startup OOM."
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
            "--max-num-batched-tokens=2048",
            "--long-prefill-token-threshold=1024",
            "--block-size=32",
            "--max-seq-len-to-capture=4096",
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
