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
        "plain_title": "Extreme Micro-Chunking with Async V2 Runner",
        "hypothesis": (
            "Crazy Idea: Obliterate head-of-line blocking by forcing extreme micro-chunking. "
            "By setting max-num-batched-tokens to 1024 and long-prefill-token-threshold to 2, "
            "we heavily chunk prefills, preventing decode stalls. We enable language-model-only "
            "to safely bypass multimodal token budget constraints at such small batch limits, "
            "and pair this with async scheduling and V2 Model Runner for relentless pipelining."
        ),
        "category": "vllm_flags",
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.92",
            "--max-num-seqs=256",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--language-model-only",
            "--max-num-batched-tokens=1024",
            "--long-prefill-token-threshold=2",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "CUDA_DEVICE_MAX_CONNECTIONS": "2",
            "OMP_NUM_THREADS": "8",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
