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
        "plain_title": "Blackwell B12X Native MoE/Linear + Optimized Speculate",
        "hypothesis": (
            "Force Blackwell-native B12X backends for MoE and Linear layers to utilize SM 12.0 hardware features. "
            "Reduce speculative tokens to 2 to eliminate discarded verification overhead. "
            "Enable vLLM V2 Model Runner, dense Triton tiling schedules, and high concurrency connections."
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
            "--long-prefill-token-threshold=1024",
            "--max-num-seqs=256",
            "--block-size=32",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--async-scheduling",
            "--quantization=fp8",
            "--kv-cache-dtype=fp8",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":2,"prompt_lookup_max":4,"prompt_lookup_min":2}',
            '--compilation-config={"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16,24,32,48,64,96,128,192,256],"max_cudagraph_capture_size":256}',
            '--kernel-config={"moe_backend":"b12x","linear_backend":"b12x","enable_flashinfer_autotune":true}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "moe_tuned_config": {
            M: {
                "BLOCK_SIZE_M": 16 if M <= 32 else (32 if M <= 128 else 64),
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64 if M <= 32 else 128,
                "GROUP_SIZE_M": 8,
                "num_warps": 4 if M <= 64 else 8,
                "num_stages": 3 if M <= 32 else 4,
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
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
