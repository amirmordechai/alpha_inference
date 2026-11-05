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
        "plain_title": "FlashInfer Chunked Prefill Symbiosis",
        "hypothesis": (
            "We pair FlashInfer's ultra-fast GQA page-attention decoding with Chunked Prefill "
            "to eliminate prefill-induced decode stalls. We lower gpu-memory-utilization to 0.90 "
            "to reserve ample scratchpad memory for FlashInfer's cooperative reductions and prevent "
            "PyTorch memory thrashing under high concurrency, while limiting max-num-batched-tokens "
            "to 4096 to regularize execution step latency."
        ),
        "category": "backends",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.90",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--language-model-only",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=4096",
            "--enable-chunked-prefill",
            "--attention-backend=FLASHINFER",
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
