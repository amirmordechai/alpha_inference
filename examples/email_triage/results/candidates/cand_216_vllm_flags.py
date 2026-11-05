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
        "plain_title": "CUDA Graph Persistence & Optimized Prefills",
        "hypothesis": (
            "We can significantly lower P99 latencies by preventing eager-mode fallbacks for long "
            "sequences through `--max-seq-len-to-capture=8192`. Additionally, we optimize chunked "
            "prefill throughput by increasing `--max-num-batched-tokens=4096`, and reduce scheduling "
            "overhead and preemption risk by lowering `--max-num-seqs=128`. We also disable periodic "
            "statistics logging to avoid I/O bottlenecks and set explicit CPU thread counts across backends."
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
            "--disable-uvicorn-access-log",
            "--disable-log-stats",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens=4096",
            "--long-prefill-token-threshold=512",
            "--max-num-seqs=128",
            "--max-seq-len-to-capture=8192",
            "--enable-prefix-caching",
            "--attention-backend=FLASHINFER",
            "--async-scheduling",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "24",
            "MKL_NUM_THREADS": "24",
            "OPENBLAS_NUM_THREADS": "24",
            "VECLIB_MAXIMUM_THREADS": "24",
            "NUMEXPR_NUM_THREADS": "24",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        },
        "custom_kernel_code": """
try:
    import gc
    gc.set_threshold(100000, 10, 10)
except Exception:
    pass

try:
    import torch
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
