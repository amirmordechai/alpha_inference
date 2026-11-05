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
        "plain_title": "Moderate Chunking (4K), Optimal Seq Count (256), Block Size 16, TF32 and GC tuning",
        "hypothesis": (
            "By scaling max-num-batched-tokens to 4096 and long-prefill-token-threshold to 1024, we reduce chunking overhead. "
            "Setting max-num-seqs to 256 avoids scheduler thrashing. Reverting block-size to 16 matches flashinfer optimal layouts, "
            "and enabling TF32 in PyTorch speeds up remaining linear ops. Disabling GC prevents latency spikes."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=5",
            "--gpu-memory-utilization=0.92",
            "--max-num-batched-tokens=4096",
            "--max-num-seqs=256",
            "--long-prefill-token-threshold=1024",
            "--block-size=16",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--attention-backend=flashinfer",
            "--async-scheduling",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "12",
            "TOKENIZERS_PARALLELISM": "true",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
        "custom_kernel_code": """
try:
    import gc
    gc.disable()
except Exception:
    pass

try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
