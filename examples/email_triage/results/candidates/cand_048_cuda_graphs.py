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
        "plain_title": "Exact-divisor 64x64 FP8 MoE tiles",
        "hypothesis": (
            "Gemma-4-26B-A4B has moe_intermediate_size=704 (11*64, not divisible by 128). "
            "Simplifying the MoE schedule to uniform 64x64 tiles eliminates K-loop tail masking "
            "and wasted padding across all token buckets."
        ),
        "category": "cuda_graphs",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=16",
            "--gpu-memory-utilization=0.94",
            "--quantization=fp8",
            "--async-scheduling",
            "--language-model-only",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=4096",
            '--compilation-config={"max_cudagraph_capture_size":512}',
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
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1 if m <= 32 else 8,
                "num_warps": 4,
                "num_stages": 4,
            }
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
