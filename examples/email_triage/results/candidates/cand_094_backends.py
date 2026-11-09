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
        "plain_title": "SM120 b12x MoE and linear kernels with FP8",
        "hypothesis": (
            "On NVIDIA RTX PRO 6000 Blackwell (SM120), fused_moe_kernel (46.9%) and linear GEMMs "
            "(26.2%) dominate GPU time. Dispatching native b12x MoE and linear backends with "
            "FlashInfer autotuning and 256-batch CUDA graphs reduces decode ITL and P99 latency."
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
            "--language-model-only",
            "--async-scheduling",
            '--kernel-config={"moe_backend":"b12x","linear_backend":"b12x","enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":256}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            m: {
                "BLOCK_SIZE_M": 16 if m <= 128 else (32 if m <= 256 else 64),
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
                "num_stages": 4,
            }
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
