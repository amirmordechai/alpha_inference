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
        "plain_title": "Drop N-Gram Spec + Exact 16x64x128 MoE Tiles",
        "hypothesis": (
            "Remove low-acceptance GPU N-gram speculation (16.9% accept) that inflates MoE decode "
            "token batches by 3x, and align uniform Triton MoE tiles to BLOCK_SIZE_N=64 (exact "
            "divisor of 704 intermediate size) to eliminate boundary masking overhead."
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
            "--language-model-only",
            "--async-scheduling",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=2048",
            "--disable-uvicorn-access-log",
            '--kernel-config={"enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":512}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": dict.fromkeys(
            (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            },
        ),
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
