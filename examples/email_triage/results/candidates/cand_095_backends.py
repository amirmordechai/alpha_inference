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
        "plain_title": "FP8 Triton MoE micro-tiles and dense CUDA graphs",
        "hypothesis": (
            "Replacing the crashing b12x config and legacy SM80 CUTLASS MoE fallback (47.4% GPU time) "
            "with FP8 quantization and the Triton MoE backend activates custom BLOCK_SIZE_M=16 micro-tiles, "
            "slashing decode ITL and P99 latency on RTX PRO 6000 Blackwell."
        ),
        "category": "backends",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=16",
            "--gpu-memory-utilization=0.95",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=4096",
            "--quantization=fp8",
            "--language-model-only",
            "--async-scheduling",
            '--kernel-config={"moe_backend":"triton"}',
            '--compilation-config={"max_cudagraph_capture_size":256,"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256]}',
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
