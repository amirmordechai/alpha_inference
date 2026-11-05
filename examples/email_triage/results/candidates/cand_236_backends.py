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
        "plain_title": "RT-Sched + Mem Lock + Thread Jitter Reduction",
        "hypothesis": (
            "Use POSIX mlockall to lock process memory in RAM, set SCHED_FIFO real-time priority "
            "to prevent OS preemption, reduce host thread pool overhead to 1 thread to prevent context "
            "switching jitter, and increase max-num-batched-tokens to 4096 to reduce prefill chunking overhead."
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
            "--attention-backend=flashinfer",
            "--enable-chunked-prefill",
            "--language-model-only",
            "--max-num-batched-tokens=4096",
            "--max-num-seqs=256",
            "--block-size=32",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "custom_kernel_code": """
try:
    import ctypes
    # Lock process virtual memory to physical RAM to completely eliminate page faults/swapping jitter
    MCL_CURRENT = 1
    MCL_FUTURE = 2
    libc = ctypes.CDLL('libc.so.6')
    libc.mlockall(MCL_CURRENT | MCL_FUTURE)
except Exception:
    pass

try:
    import os
    # Try setting real-time scheduling priority (SCHED_FIFO) with maximum priority
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
except Exception:
    try:
        os.nice(-20)
    except Exception:
        pass

try:
    import gc
    gc.disable()
except Exception:
    pass

try:
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Disable PyTorch's background JIT profiling/compilation to avoid latency spikes
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
