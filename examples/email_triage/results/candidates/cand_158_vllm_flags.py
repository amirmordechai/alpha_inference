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
        "plain_title": "The Naked Truth: Pure Continuous Batching",
        "hypothesis": (
            "Strip away algorithmic overheads (chunked prefill, prefix caching, async scheduling) "
            "and messy hacky patches. Rely purely on unadulterated, standard continuous batching "
            "with a large batch token size to drastically reduce CPU scheduling latency."
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
            "--max-num-batched-tokens=16384",
            "--max-num-seqs=128",
            "--block-size=16",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
