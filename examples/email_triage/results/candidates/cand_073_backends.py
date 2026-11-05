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
        "plain_title": "Optimized Memory & CPU Thread Concurrency Tuning",
        "hypothesis": (
            "Reduce max-num-batched-tokens to 4096 and max-num-seqs to 128 to optimize prefill chunk size and scheduler overhead, "
            "adjust GPU memory utilization to 0.90 for PyTorch allocator headroom, tune CPU threads to 4, "
            "and use relaxed GC thresholds with explicit PyTorch thread / TF32 tuning."
        ),
        "category": "backends",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.90",
            "--attention-backend=flashinfer",
            "--enable-chunked-prefill",
            "--long-prefill-token-threshold=512",
            "--language-model-only",
            "--max-num-batched-tokens=4096",
            "--max-num-seqs=128",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
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
import torch
try:
    gc.enable()
    gc.set_threshold(50000, 500, 500)
except Exception:
    pass
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
