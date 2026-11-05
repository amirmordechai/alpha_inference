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
        "plain_title": "Optimized Scheduler & CUDA Graph Capture",
        "hypothesis": (
            "Tune scheduler concurrency (max-num-seqs=128, max-num-batched-tokens=4096, long-prefill-token-threshold=1024) "
            "to prevent decode starvation and step latency spikes. Extend CUDA graph capturing up to 4096 tokens "
            "to keep long-sequence decodes in fast-path, and enable PyTorch reduced precision reduction settings."
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
            "--enable-chunked-prefill",
            "--long-prefill-token-threshold=1024",
            "--language-model-only",
            "--max-num-batched-tokens=4096",
            "--max-num-seqs=128",
            "--enable-prefix-caching",
            "--disable-uvicorn-access-log",
            "--attention-backend=flashinfer",
            "--async-scheduling",
            "--max-seq-len-to-capture=4096",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "TORCH_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        },
        "custom_kernel_code": """
import gc
import torch
try:
    gc.disable()
except Exception:
    pass
try:
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
