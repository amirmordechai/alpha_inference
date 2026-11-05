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
        "plain_title": "Chunked Prefill (2048) + GC and Concurrency Tuning",
        "hypothesis": (
            "Enable real chunked prefill by reducing max-num-batched-tokens to 2048, which prevents long "
            "prefill pauses and improves P99 latency. Increase max-num-seqs to 256 to handle high arrival "
            "rates. Optimize GC thresholds to prevent cyclic collection spikes, set OMP_NUM_THREADS to 8, "
            "and set CUDA_DEVICE_MAX_CONNECTIONS to 16 for better overlap."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.92",
            "--language-model-only",
            "--max-num-batched-tokens=2048",
            "--max-num-seqs=256",
            "--enable-chunked-prefill",
            "--long-prefill-token-threshold=1024",
            "--async-scheduling",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--attention-backend=flashinfer",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "16",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "custom_kernel_code": """
try:
    import torch
    import gc
    
    # Enable TF32 for any FP32 fallbacks
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Disable PyTorch JIT profiling executor spikes
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    
    # Tune GC thresholds to reduce frequency of GC overhead in runtime loops
    gc.set_threshold(50000, 15, 15)
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
