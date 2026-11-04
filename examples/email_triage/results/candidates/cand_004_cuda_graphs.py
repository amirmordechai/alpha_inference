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
        "plain_title": "Optimized batching, CUDA graphs and compilation",
        "hypothesis": (
            "Tuning micro-batch limits (max-num-seqs=10, max-num-batched-tokens=4096) to "
            "reduce P99 queueing delay, lowering logprobs overhead, maximizing memory utilization "
            "to 0.95, and leveraging aggressive torch.compile / Inductor optimization settings."
        ),
        "category": "cuda_graphs",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
            "--served-model-name=diffusiongemma-26b",
            "--host=0.0.0.0",
            "--port=8080",
            '--diffusion-config={"canvas_length": 64}',
            "--max-logprobs=5",
            '--override-generation-config={"max_new_tokens": null}',
            "--enable-auto-tool-choice",
            "--tool-call-parser=gemma4",
            "--reasoning-parser=gemma4",
            "--enable-prefix-caching",
            "--async-scheduling",
            '--limit-mm-per-prompt={"image": 0, "video": 0}',
            "--attention-backend=TRITON_ATTN",
            "--max-num-seqs=10",
            "--max-num-batched-tokens=4096",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--disable-uvicorn-access-log",
            '--compilation-config={"max_cudagraph_capture_size": 2048}',
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "PYTHONUNBUFFERED": "1",
        },
        "custom_kernel_code": """try:
    import torch
    import torch._inductor.config as config
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.triton.cooperative_reduction = True
    config.triton.unique_kernel_names = True
    config.force_fuse_int_mm_with_mul = True
    config.fx_graph_cache = True
    config.compile_threads = 16
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
