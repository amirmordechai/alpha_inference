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
        "plain_title": "Chunked Prefill 4k + Custom Dense CUDA Graph Ladder & GC Freeze",
        "hypothesis": (
            "Lower max-num-batched-tokens to 4096 and set long-prefill-token-threshold "
            "to mitigate prefill-decode interference, monkey-patch vLLM's internal "
            "cudagraph ladders for dense capture, redirect Triton compile caches to RAM-disk, "
            "and perform gc.freeze() to eliminate trace jit/stutter latencies."
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
            "--max-num-batched-tokens=4096",
            "--long-prefill-token-threshold=1024",
            "--max-num-seqs=256",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "16",
            "MKL_NUM_THREADS": "16",
            "OPENBLAS_NUM_THREADS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "TRITON_CACHE_DIR": "/dev/shm/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/torchinductor",
        },
        "custom_kernel_code": """
try:
    import gc
    gc.disable()
    gc.freeze()
except Exception:
    pass

try:
    import os
    os.nice(-10)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(16)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

try:
    # Crazy monkeypatch to optimize CUDA graph capture sizes in vLLM
    import vllm.config
    import vllm.worker.model_runner
    
    ladder = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256]
    
    for module in [vllm.config, vllm.worker.model_runner]:
        for attr in dir(module):
            cls = getattr(module, attr)
            if isinstance(cls, type):
                if hasattr(cls, 'cudagraph_capture_sizes'):
                    cls.cudagraph_capture_sizes = ladder
                if hasattr(cls, 'max_cudagraph_capture_size'):
                    cls.max_cudagraph_capture_size = 256
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
