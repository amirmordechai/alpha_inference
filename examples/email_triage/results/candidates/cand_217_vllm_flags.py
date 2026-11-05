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
        "plain_title": "Balanced Chunking, Prefix Caching, and FlashAttention",
        "hypothesis": (
            "Extreme micro-chunking degraded throughput and worsened queueing delay. "
            "Relax to --max-num-batched-tokens=4096 and --max-num-seqs=128. "
            "Remove V2 model runner to simplify execution. Switch to flash_attn "
            "backend and use a 2048 token threshold."
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
            "--max-num-batched-tokens=4096",
            "--max-num-seqs=128",
            "--long-prefill-token-threshold=2048",
            "--block-size=16",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--attention-backend=flash_attn",
            "--async-scheduling",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "CUDA_DEVICE_MAX_CONNECTIONS": "2",
        },
        "custom_kernel_code": """
try:
    import gc
    gc.disable()
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
