"""Seed inference configuration evolved by Google Cloud AlphaEvolve (`candidate.py`).

AlphaEvolve mutates the function inside the evolve block below to discover faster
vLLM scheduler flags, CUDA graph capture ladders, runtime backends, and custom
Triton GPU kernels tailored to your workload trace and GPU hardware.
"""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
# IMPORTANT: Keep the function name `get_candidate_config` and return dict keys unchanged.
def get_candidate_config() -> dict[str, Any]:
    """Return the vLLM serving configuration and optional custom Triton GPU kernel."""
    return {
        "plain_title": "High-Throughput Standard Prefill",
        "hypothesis": (
            "The massive P99 TTFT matching P99 overall latency indicates heavy queueing or preemptions under load. "
            "Chunked prefill explicitly delays TTFT for long prompts by splitting them. By returning to standard prefill "
            "with a massive max_num_batched_tokens limit (16384), we allow long prefills to finish in one pass. Lowering "
            "max_num_seqs to 128 mitigates KV cache exhaustion and preemption recomputes on the 48GB GPU."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.95",
            "--language-model-only",
            "--max-num-batched-tokens=16384",
            "--max-num-seqs=128",
            "--async-scheduling",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--attention-backend=flashinfer",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "4",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "custom_kernel_code": """
try:
    import torch
    
    # Disable PyTorch JIT profiling executor spikes
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    
    # Enable TF32 for any FP32 fallbacks
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
