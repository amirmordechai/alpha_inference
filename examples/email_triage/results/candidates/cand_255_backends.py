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
        "plain_title": "Aggressive Prefill, Block 16, & Sync-Free CUDA Allocator",
        "hypothesis": (
            "Increase max_num_seqs to 256 and max_num_batched_tokens to 8192 with block_size=16 "
            "to maximize FlashInfer execution efficiency. Monkey patch torch.cuda.empty_cache "
            "to completely eliminate expensive GPU synchronization pauses during execution."
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
            "--disable-uvicorn-access-log",
            "--language-model-only",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--attention-backend=flashinfer",
            "--async-scheduling",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=2048",
            "--block-size=16",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "custom_kernel_code": """
try:
    import gc
    import sys
    # Prevent GIL thrashing between Uvicorn and model executor threads
    sys.setswitchinterval(1.0)
    # Completely disable Garbage Collection during inference
    gc.disable()
    # Disable autograd logging and anomaly tracking overhead
    import torch
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    # Enable aggressive matmul and convolution micro-optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Monkey patch empty_cache and ipc_collect to avoid expensive global syncs
    torch.cuda.empty_cache = lambda *args, **kwargs: None
    torch.cuda.ipc_collect = lambda *args, **kwargs: None
    # Optimize PyTorch threading setup
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
