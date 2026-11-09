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
        "plain_title": "FP8 + Ngram-GPU Spec + Uniform M16N128 MoE",
        "hypothesis": (
            "Restoring ngram_gpu (k=3) cuts sequential decode steps while using a uniform "
            "(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, num_stages=4) MoE tile config "
            "across all M buckets eliminates mid-trace Triton JIT specialization stalls."
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
            "--max-num-batched-tokens=4096",
            "--quantization=fp8",
            "--language-model-only",
            "--async-scheduling",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            '--kernel-config={"enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":512}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            m: {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
