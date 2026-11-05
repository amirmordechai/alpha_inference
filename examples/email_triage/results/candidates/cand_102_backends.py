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
        "plain_title": "Stable V0 Engine + LM Only + FlashInfer",
        "hypothesis": (
            "Disable the experimental V1 engine by removing VLLM_USE_V2_MODEL_RUNNER to resolve "
            "startup crashes. Enable --language-model-only for this multimodal checkpoint "
            "to optimize for text-only traces and support lower batched token sizes. Scale "
            "max-num-seqs to 128 to prevent queue starvation under concurrency."
        ),
        "category": "backends",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.95",
            "--enable-prefix-caching",
            "--max-num-seqs=128",
            "--enable-chunked-prefill",
            "--language-model-only",
            "--max-num-batched-tokens=4096",
            "--attention-backend=FLASHINFER",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
