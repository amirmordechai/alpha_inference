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
        "plain_title": "Maximized Concurrency & Batch Sizes",
        "hypothesis": (
            "Increase concurrency by raising max-num-seqs to 256 and max-num-batched-tokens to 16384, "
            "tuning long-prefill-token-threshold to 2048 to balance prefill and decode phases. "
            "Removed O(N^2) dynamic import hook which caused catastrophic request queueing."
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
            "--max-num-batched-tokens=16384",
            "--max-num-seqs=256",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--attention-backend=FLASHINFER",
            "--block-size=16",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "4",
            "TORCH_NUM_THREADS": "4",
            "CUDA_DEVICE_MAX_CONNECTIONS": "2",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
        },
        "custom_kernel_code": """
import gc
import sys
import os
import torch

try:
    gc.disable()
except Exception:
    pass

try:
    # Use default or slightly tuned switch interval to allow responsive async execution
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
""",
    }
# EVOLVE-BLOCK-END
