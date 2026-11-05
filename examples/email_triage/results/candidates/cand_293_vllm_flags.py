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
        "plain_title": "Tune Concurrency and Batching for P99",
        "hypothesis": (
            "Optimize P99 latency by lowering max-num-seqs to 128 to prevent queue overloading "
            "and increasing gpu-memory-utilization to 0.95 to maximize KV cache. We also set "
            "max-num-batched-tokens to 8192 and long-prefill-token-threshold to 2048 to balance "
            "chunked prefill efficiency."
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
            "--enable-chunked-prefill",
            "--long-prefill-token-threshold=2048",
            "--language-model-only",
            "--max-num-batched-tokens=8192",
            "--max-num-seqs=128",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--attention-backend=flashinfer",
            "--block-size=16",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "4",
            "TORCH_NUM_THREADS": "4",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "custom_kernel_code": """
import gc
import sys
import os
import torch
import builtins

try:
    gc.disable()
except Exception:
    pass

try:
    # Relax Python thread switching to significantly reduce GIL contention on background serving threads
    sys.setswitchinterval(0.005)
except Exception:
    pass

try:
    # Pin process to available CPU cores for better cache locality and predictable execution
    if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
        all_cpus = os.sched_getaffinity(0)
        os.sched_setaffinity(0, all_cpus)
except Exception:
    pass

try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except Exception:
    pass

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# Removed import hook monkey-patch to avoid overhead and instability.
""",
    }
# EVOLVE-BLOCK-END
