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
        "plain_title": "Chunked Prefill + UVLoop Event Loop Optimization",
        "hypothesis": (
            "Crazy Idea: To address the massive head-of-line prefill-decode interference "
            "on the 26B model, we enable `--enable-chunked-prefill` with smaller batched "
            "tokens (4096) and a threshold of 1024 to split long prefills. This prevents decode "
            "starvation and drives P99 down. We also inject `uvloop` as the global asyncio event loop "
            "via `sitecustomize.py` to optimize host event loop scheduling, remove the buggy experimental "
            "V2 model runner, and tune OMP threads and CUDA connections to eliminate driver overhead."
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
            "--language-model-only",
            "--disable-uvicorn-access-log",
            "--max-num-batched-tokens=4096",
            "--max-num-seqs=256",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--long-prefill-token-threshold=1024",
            "--attention-backend=FLASHINFER",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "12",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
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
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
