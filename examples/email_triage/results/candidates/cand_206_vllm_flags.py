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
        "plain_title": "Async Micro-Burst Scheduling & Core Saturation",
        "hypothesis": (
            "Crazy Idea: Decouple the serving event loop completely using `--async-scheduling` while "
            "hyper-tuning Chunked Prefill with `--max-num-batched-tokens=2048` and `--long-prefill-token-threshold=512` "
            "to force micro-prefill bursts that absolutely never block active decodes. "
            "Furthermore, we drastically crank `OMP_NUM_THREADS` to `24` to fully saturate the host's 48 CPUs "
            "during async orchestration, obliterating P99 tail latencies caused by CPU scheduling stalls."
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
            "--enable-chunked-prefill",
            "--max-num-batched-tokens=2048",
            "--long-prefill-token-threshold=512",
            "--max-num-seqs=256",
            "--enable-prefix-caching",
            "--attention-backend=FLASHINFER",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "24",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        },
        "custom_kernel_code": """
try:
    import gc
    gc.disable()
except Exception:
    pass

try:
    import torch
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
