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
        "plain_title": "Optimized Chunked Prefill & CUDA Graphs",
        "hypothesis": (
            "Increase max-num-batched-tokens to 8192 to speed up prefills and reduce TTFT, "
            "tune max-num-seqs to 192 to prevent preemption, and add customized "
            "compilation config for CUDA graph capture."
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
            "--enable-chunked-prefill",
            "--language-model-only",
            "--max-num-batched-tokens=8192",
            "--max-num-seqs=192",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--disable-log-stats",
            "--async-scheduling",
            "--long-prefill-token-threshold=1024",
            '--compilation-config={"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,192],"max_cudagraph_capture_size":512}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "4",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
