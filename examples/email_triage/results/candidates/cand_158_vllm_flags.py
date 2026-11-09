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
        "plain_title": "GPU N-gram speculation with tiered FP8 MoE tiles",
        "hypothesis": (
            "Pairing zero-VRAM ngram_gpu speculative decoding (3 tokens) with a 512-token CUDA graph "
            "ladder and tiered FP8 Triton MoE tiles cuts sequential decode steps and MoE kernel overhead."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=16",
            "--gpu-memory-utilization=0.95",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=8192",
            "--quantization=fp8",
            "--language-model-only",
            "--async-scheduling",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            '--kernel-config={"moe_backend":"triton","enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":512,"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256,320,384,448,512]}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            **dict.fromkeys(
                (1, 2, 4, 8, 16, 32, 64),
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 4,
                },
            ),
            **dict.fromkeys(
                (128, 256),
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                    "num_warps": 4,
                    "num_stages": 4,
                },
            ),
            512: {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
                "num_stages": 4,
            },
            1024: {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_warps": 8,
                "num_stages": 4,
            },
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
