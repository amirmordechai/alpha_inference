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
        "plain_title": "Optimize Concurrency Limits and V2 Runner",
        "hypothesis": (
            "Reducing --max-num-seqs to 128 caps concurrency to avoid decoding stalls. "
            "Increasing --max-num-batched-tokens to 16384 and --long-prefill-token-threshold to 4096 "
            "increases arithmetic intensity for large prefills. Enabling VLLM_USE_V2_MODEL_RUNNER "
            "leverages the newer C++ scheduler loop for reduced host overhead."
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
            "--max-num-batched-tokens=16384",
            "--max-num-seqs=128",
            "--long-prefill-token-threshold=4096",
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
            "VLLM_USE_V2_MODEL_RUNNER": "1",
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
