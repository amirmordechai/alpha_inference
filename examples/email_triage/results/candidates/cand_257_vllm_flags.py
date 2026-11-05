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
        "plain_title": "Balanced Sequence Chunking and GC Tuning",
        "hypothesis": (
            "Relax to --max-num-seqs=256 and --max-num-batched-tokens=8192 for balanced sequence-dimension blocking "
            "(recovering arithmetic intensity lost in extreme micro-chunking). Disable V2 runner to avoid regressions, "
            "and tune Python GC thresholds instead of fully disabling it to prevent out-of-memory or synchronous stalls."
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
            "--max-num-batched-tokens=8192",
            "--max-num-seqs=256",
            "--long-prefill-token-threshold=2048",
            "--block-size=16",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--attention-backend=flashinfer",
            "--async-scheduling",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
        },
        "custom_kernel_code": """
try:
    import gc
    gc.set_threshold(500000, 250, 250)
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
