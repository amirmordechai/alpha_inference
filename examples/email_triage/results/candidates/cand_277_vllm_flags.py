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
        "plain_title": "Extreme Batching Asymmetry & TTFT Annihilation",
        "hypothesis": (
            "Crazy Idea: The trace shows TTFT_P99 is exactly equal to P99 latency (8588ms). "
            "Small prefill chunks (2048) are catastrophically bottlenecking large prompts. "
            "We annihilate TTFT by scaling `--max-num-batched-tokens` to `16384` and dropping "
            "async scheduling/chunking flags to force monolithic, hyper-fast prefill passes. "
            "Simultaneously, we restrict `--max-num-seqs` to `128` to strictly limit contention and "
            "decode starvation. OMP_NUM_THREADS is reduced to 8 for optimal host thread scheduling."
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
            "--max-num-batched-tokens=16384",
            "--max-num-seqs=128",
            "--enable-prefix-caching",
            "--attention-backend=FLASHINFER",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "4",
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
