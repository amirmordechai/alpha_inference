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
        "plain_title": "Block-32 + Dynamic Chunk Threshold + Async Concurrency Optimization",
        "hypothesis": (
            "We aggressively optimize both memory layout and scheduler timing. First, we increase KV "
            "block size to 32 to align with FlashInfer's optimal memory access layout and reduce page table "
            "size on the GPU. Second, we raise max batched tokens to 8192 and the long prefill threshold to "
            "2048 to prevent unnecessary chunking of medium-sized prefills, while limiting max concurrency "
            "to 128 to minimize queueing delay. Third, we set CUDA_DEVICE_MAX_CONNECTIONS to 32 to maximize "
            "stream overlapping, increase CPU threads to 16, enable TF32 on the GPU, and optimize the "
            "PyTorch allocator with max_split_size_mb:512."
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
            "--disable-uvicorn-access-log",
            "--max-num-batched-tokens=8192",
            "--max-num-seqs=128",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--long-prefill-token-threshold=2048",
            "--attention-backend=FLASHINFER",
            "--block-size=32",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "16",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        },
        "custom_kernel_code": """
try:
    import gc
    gc.disable()
except Exception:
    pass

try:
    import uvloop
    import asyncio
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except Exception:
    pass

try:
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
except Exception:
    pass

try:
    import resource
    nofile_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, nofile_limit[1]))
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
