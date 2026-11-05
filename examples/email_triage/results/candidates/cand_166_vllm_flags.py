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
        "plain_title": "Chunked Prefill & Latency-Optimal Scheduling",
        "hypothesis": (
            "Enable chunked prefill with a chunk size of 4096 to prevent long prefills from "
            "starving decode steps, dramatically reducing P99 latency. Couple this with "
            "CUDA stream serialization and disabling latent PyTorch profiling overhead to "
            "ensure maximum hardware execution efficiency."
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
            "--disable-uvicorn-access-log",
            "--language-model-only",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--attention-backend=flashinfer",
            "--async-scheduling",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=4096",
            "--long-prefill-token-threshold=2048",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
        "custom_kernel_code": """
try:
    import torch
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
