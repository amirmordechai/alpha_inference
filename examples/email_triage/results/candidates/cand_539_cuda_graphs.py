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
    """Return the vLLM serving configuration and optional custom Triton GPU kernel."""
    return {
        "plain_title": "Minimalist High-Throughput Core",
        "hypothesis": (
            "Crazy simplification: Strip out all speculative decoding, chunked prefill, prefix caching, and custom kernel hacks. "
            "Rely entirely on raw fp8 compute throughput, maximizing max-num-batched-tokens to 8192, and using pure "
            "FlashInfer + Triton MoE without overhead from extra feature complexity."
        ),
        "category": "cuda_graphs",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.95",
            "--attention-backend=flashinfer",
            "--quantization=fp8",
            "--kv-cache-dtype=fp8",
            '--kernel-config={"moe_backend":"triton","linear_backend":"auto","enable_flashinfer_autotune":true}',
            '--compilation-config={"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256],"max_cudagraph_capture_size":256}',
            "--language-model-only",
            "--max-num-batched-tokens=8192",
            "--max-num-seqs=256",
            "--block-size=32",
            "--disable-uvicorn-access-log",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "moe_tuned_config": {
            M: {
                "BLOCK_SIZE_M": max(16, min(M, 64)),
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 16,
                "num_warps": 4 if M <= 128 else 8,
                "num_stages": 4
            }
            for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
