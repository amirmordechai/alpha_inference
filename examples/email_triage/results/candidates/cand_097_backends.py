"""Seed inference configuration evolved by Google Cloud AlphaEvolve (`candidate.py`).

AlphaEvolve mutates the function inside the evolve block below to discover faster
vLLM scheduler flags, CUDA graph capture ladders, runtime backends, and custom
Triton GPU kernels tailored to your workload trace and GPU hardware.
"""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
# IMPORTANT: Keep the function name `get_candidate_config`.
# Supported dict keys: plain_title, hypothesis, category, vllm_args, env_vars, moe_tuned_config, custom_kernel_code.
def get_candidate_config() -> dict[str, Any]:
    """Return the vLLM serving configuration, optional MoE tile schedule, and optional Triton GPU kernel."""
    return {
        "plain_title": "FP8 with CUTLASS MoE backend and async scheduling",
        "hypothesis": (
            "Under online FP8 quantization, dense projections use native SM120 CUTLASS kernels (16.8% GPU) "
            "while MoE falls back to Triton fused_moe_kernel (46.5% GPU). Dispatching MoE to CUTLASS "
            "with FlashInfer autotuning and async scheduling targets the primary decode bottleneck."
        ),
        "category": "backends",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=16",
            "--gpu-memory-utilization=0.92",
            "--quantization=fp8",
            "--kernel-config={\"moe_backend\":\"cutlass\",\"enable_flashinfer_autotune\":true}",
            "--async-scheduling",
            "--language-model-only",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {},
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
