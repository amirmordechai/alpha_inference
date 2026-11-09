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
        "plain_title": "FP8 + 2-Tok N-Gram + M16 Sparse MoE Tiles",
        "hypothesis": (
            "Trimming ngram_gpu speculation from 3 to 2 tokens cuts verification batch "
            "inflation by 25% at 12.6% acceptance, while holding BLOCK_SIZE_M=16 through "
            "M=256 matches actual E=128 top_k=8 density (M/16 <= 16 tokens/expert)."
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
            "--max-num-batched-tokens=4096",
            "--disable-uvicorn-access-log",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":2,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            '--compilation-config={"max_cudagraph_capture_size":512}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            m: {
                "BLOCK_SIZE_M": 16 if m <= 256 else (32 if m <= 512 else 64),
                "BLOCK_SIZE_N": 64 if m <= 32 else 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1 if m <= 16 else 8,
                "num_warps": 4 if m <= 128 else 8,
                "num_stages": 4,
            }
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
