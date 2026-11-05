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
        "plain_title": "Chunked Prefill with High-Concurrency CUDA Graphs",
        "hypothesis": (
            "Enabling chunked prefill with max-num-batched-tokens=2048 eliminates prefill-decode "
            "interference by scheduling prefills in small chunks, directly reducing tail latency. "
            "By setting VLLM_DEV_CUDA_GRAPHS_MAX_BATCH_SIZE=128, we prevent eager-mode fallback under "
            "high concurrency. Reverting V2 model runner avoids experimental overhead, and FLASH_ATTN "
            "with block-size=16 optimizes execution on Ada Lovelace SM 8.9."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.90",
            "--language-model-only",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens=2048",
            "--max-num-seqs=128",
            "--block-size=16",
            "--async-scheduling",
            "--attention-backend=FLASH_ATTN",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "VLLM_DEV_CUDA_GRAPHS_MAX_BATCH_SIZE": "128",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": "60",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
