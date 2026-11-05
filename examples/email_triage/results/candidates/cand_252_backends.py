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
        "plain_title": "Balanced GIL Switching, Block Size 32 & Parallel CUDA Connections",
        "hypothesis": (
            "Fix severe GIL starvation of the Uvicorn HTTP server by reducing the switch interval "
            "from 500ms to 15ms. Upgrade block_size to 32 to reduce scheduling metadata tracking "
            "overhead and optimize FlashInfer alignment. Increase CUDA connections to 32 to enable "
            "kernel overlap, and set jemalloc dirty decay to prevent page-fault stutters."
        ),
        "category": "backends",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.92",
            "--disable-uvicorn-access-log",
            "--language-model-only",
            "--enable-prefix-caching",
            "--attention-backend=flashinfer",
            "--async-scheduling",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=4096",
            "--long-prefill-token-threshold=1024",
            "--block-size=32",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "MALLOC_CONF": "background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000",
        },
        "custom_kernel_code": """
try:
    import gc
    import sys
    # Set a balanced GIL switch interval to prevent Uvicorn starvation
    sys.setswitchinterval(0.015)
    # Defer Garbage Collection cycles from the critical inference steps
    gc.set_threshold(100000, 2000, 2000)
    # Disable autograd logging and anomaly tracking overhead
    import torch
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
