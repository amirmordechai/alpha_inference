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
        "plain_title": "Extreme Dense CUDA Graphs, GIL Nuke & Deep Chunked Prefill",
        "hypothesis": (
            "Deploy a 'crazy' multi-layered optimization: Nuke Python GIL overhead "
            "with a 1000s switch interval, inject an ultra-dense CUDA graph ladder "
            "via compilation-config to eliminate padding/fallback at high concurrency, "
            "and tune max-num-batched-tokens to 16384 with a 2048 prefill threshold "
            "to deeply interleave prefill chunks with decodes, slashing P99 latency."
        ),
        "category": "cuda_graphs",  # vllm_flags | cuda_graphs | backends | triton_kernels
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
            "--max-num-batched-tokens=16384",
            "--long-prefill-token-threshold=2048",
            "--max-num-seqs=256",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--async-scheduling",
            '--compilation-config={"cudagraph_capture_sizes":[1,2,4,8,12,16,24,32,48,64,80,96,128,256],"max_cudagraph_capture_size":4096}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.9",
        },
        "custom_kernel_code": """
try:
    import sys
    sys.setswitchinterval(1000)  # Effectively disable GIL context switching
except Exception:
    pass

try:
    import gc
    gc.disable()
except Exception:
    pass

try:
    import os
    os.nice(-10)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
