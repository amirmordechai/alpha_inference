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
        "plain_title": "Optimized Concurrency, Larger Batched Tokens & Increased KV Cache",
        "hypothesis": (
            "Reduce max-num-seqs to 128 to prevent decoding step latency bloat under load. "
            "Increase max-num-batched-tokens to 8192 for higher prefill throughput, and "
            "raise GPU memory utilization to 0.95 to maximize KV cache block capacity. "
            "Adjust GIL switch interval and OMP threads for better background host parallelism."
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
            "--disable-uvicorn-access-log",
            "--language-model-only",
            "--enable-prefix-caching",
            "--attention-backend=flashinfer",
            "--async-scheduling",
            "--max-num-seqs=128",
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=4096",
            "--block-size=16",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "4",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "MALLOC_CONF": "background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000",
        },
        "custom_kernel_code": """
try:
    import gc
    import sys
    # Set an aggressive GIL switch interval to maintain fast API responsiveness
    sys.setswitchinterval(0.005)
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
