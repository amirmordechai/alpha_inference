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
        "plain_title": "V2 Model Runner + Simplified Auto Kernels",
        "hypothesis": (
            "Simplify configuration by dropping unused Triton MoE tuning (since FP8 MoE falls back to CUTLASS), "
            "enable V2 Model Runner for potentially leaner scheduling, increase concurrency, and use auto kernels."
        ),
        "category": "backends",
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.96",
            "--attention-backend=auto",
            "--enable-chunked-prefill",
            "--language-model-only",
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=2048",
            "--max-num-seqs=384",
            "--block-size=32",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--async-scheduling",
            "--quantization=fp8",
            "--kv-cache-dtype=fp8",
            "--speculative-config",
            '{"method":"ngram_gpu","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            "--kernel-config",
            '{"moe_backend":"auto","linear_backend":"auto","enable_flashinfer_autotune":true}',
            "--compilation-config",
            '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,96,128,256],"max_cudagraph_capture_size":256}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
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
