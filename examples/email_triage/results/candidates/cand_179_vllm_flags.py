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
        "plain_title": "Quantum Micro-Batching & FlashInfer",
        "hypothesis": (
            "Crazy Idea: Slice prefill chunks into ultra-tiny 512-token micro-batches to create a 'quantum' "
            "overlap between prefill and decode, completely eradicating prefill latency spikes. Combined with "
            "FlashInfer, massive concurrency (256 seqs), and stripping legacy V2 env vars to ensure V1 engine "
            "stability, this aggressively trades max throughput for the absolute lowest P99 tail latency possible."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.90",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--attention-backend=FLASHINFER",
            "--language-model-only",
            "--enable-chunked-prefill",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=512",
            "--long-prefill-token-threshold=512",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
