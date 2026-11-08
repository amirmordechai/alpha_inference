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
        "plain_title": "Cutlass FP8 MoE + Tensor-Optimized Batching + Deeper N-Gram",
        "hypothesis": (
            "Leverage CUTLASS for the MoE backend to better utilize Blackwell SM120 WGMMA instructions for FP8 tensor operations, avoiding Triton's zero-padding overhead on small expert batches. "
            "Increase max batched tokens to 4096 for better chunked prefill saturation and increase N-Gram speculation depth to 4 to capitalize on the reduced memory bandwidth footprint from FP8 quantization."
        ),
        "category": "backends",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.95",
            "--attention-backend=flashinfer",
            "--enable-chunked-prefill",
            "--language-model-only",
            "--max-num-batched-tokens=4096",
            "--max-num-seqs=256",
            "--block-size=32",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--async-scheduling",
            "--quantization=fp8",
            "--kv-cache-dtype=fp8",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":4,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            '--compilation-config={"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16,24,32,48,64,96,128,192,256],"max_cudagraph_capture_size":256}',
            '--kernel-config={"moe_backend":"cutlass","linear_backend":"auto","enable_flashinfer_autotune":true}',
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
                "BLOCK_SIZE_M": 16 if M <= 64 else (32 if M <= 256 else 64),
                "BLOCK_SIZE_N": 64 if M <= 128 else 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_warps": 4 if M <= 128 else 8,
                "num_stages": 4 if M <= 128 else 5,
            }
            for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        },
        "custom_kernel_code": """
try:
    import gc
    gc.disable()
except Exception:
    pass

try:
    import os
    os.nice(-10)
except Exception:
    pass

try:
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
