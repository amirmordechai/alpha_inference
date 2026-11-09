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
        "plain_title": "Drop ngram_gpu overflow & tune regime MoE tiles",
        "hypothesis": (
            "2-token ngram_gpu tripled active decode tokens per step (M~90 -> M~270), pushing "
            "tokens/expert past 16 and forcing 2x BLOCK_SIZE_M=16 MoE tiles for only 16.9% "
            "acceptance. Dropping ngram_gpu keeps decode within 1 tile/expert while scaling "
            "M>=256 prefill buckets to BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, num_warps=8."
        ),
        "category": "cuda_graphs",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=16",
            "--gpu-memory-utilization=0.95",
            "--quantization=fp8",
            "--language-model-only",
            "--async-scheduling",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=8192",
            '--compilation-config={"max_cudagraph_capture_size":512}',
            '--kernel-config={"enable_flashinfer_autotune":true}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
        },
        "moe_tuned_config": {
            **{
                m: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4}
                for m in (1, 2, 4, 8, 16, 32, 64, 128)
            },
            **{
                m: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4}
                for m in (256, 512, 1024)
            },
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
