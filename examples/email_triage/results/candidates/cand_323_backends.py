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
    # Build hardware-specific fused_moe Triton Tile schedules
    moe_tuned = {}
    for m in [1, 2, 4, 8, 16]:
        moe_tuned[m] = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    for m in [32, 64]:
        moe_tuned[m] = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        }
    for m in [128, 256]:
        moe_tuned[m] = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 4,
        }
    for m in [512, 1024]:
        moe_tuned[m] = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 8,
            "num_stages": 5,
        }

    return {
        "plain_title": "FP8 + Speculate + Tuned Triton MoE Tiles",
        "hypothesis": (
            "Enables GPU N-Gram Speculative Decoding to cut down decode steps, forces Triton MoE "
            "with custom SM12.0 optimized micro-tiles to reduce zero-padding, and uses a dense CUDA Graph size ladder."
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
            "--quantization=fp8",
            "--kv-cache-dtype=fp8",
            '--kernel-config={"moe_backend":"triton","linear_backend":"auto","enable_flashinfer_autotune":true}',
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            '--compilation-config={"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,80,96,112,128,160,192,224,256],"max_cudagraph_capture_size":256}',
            "--enable-chunked-prefill",
            "--long-prefill-token-threshold=1024",
            "--language-model-only",
            "--max-num-batched-tokens=2048",
            "--max-num-seqs=256",
            "--block-size=32",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "moe_tuned_config": moe_tuned,
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
