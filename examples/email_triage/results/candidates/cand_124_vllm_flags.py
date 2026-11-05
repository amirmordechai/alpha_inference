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
        "plain_title": "LM-Only Chunked Prefill + V2 Runner + FlashInfer",
        "hypothesis": (
            "Enabling `--enable-chunked-prefill` along with tuning `--long-prefill-token-threshold=2048` "
            "prevents long prefill requests from starving or interrupting ongoing decodes, which directly "
            "improves P99 latency. We also set `CUDA_DEVICE_MAX_CONNECTIONS=1` to minimize driver scheduling overhead."
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
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--long-prefill-token-threshold=2048",
            "--max-num-seqs=256",
            "--language-model-only",
            "--max-num-batched-tokens=8192",
            "--disable-uvicorn-access-log",
            "--attention-backend=FLASHINFER",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
