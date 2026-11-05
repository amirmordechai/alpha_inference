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
        "plain_title": "FlashInfer Backend with Single-Step Prefill & V2 Runner",
        "hypothesis": (
            "Disable chunked prefill and set max batched tokens to 8192 to minimize TTFT. "
            "Enable FlashInfer attention backend for state-of-the-art Gemma attention execution. "
            "Activate V2 Model Runner and optimize OMP threads to minimize CPU host overhead."
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
            "--max-num-batched-tokens=8192",
            "--attention-backend=flashinfer",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
