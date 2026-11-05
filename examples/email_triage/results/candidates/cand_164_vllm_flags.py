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
        "plain_title": "Simplified Config + Chunked Prefill",
        "hypothesis": (
            "Remove the complex god-mode hacks, GC disabling, and thread bindings which over-complicate "
            "and potentially harm performance. Instead, simplify the configuration and use vLLM's native "
            "chunked prefill and async scheduling to balance TTFT and decode latency cleanly."
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
            "--language-model-only",
            "--max-num-batched-tokens=8192",
            "--max-num-seqs=256",
            "--enable-chunked-prefill",
            "--async-scheduling",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--attention-backend=flashinfer",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "custom_kernel_code": """
try:
    import torch
    
    # Disable PyTorch JIT profiling executor spikes
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
