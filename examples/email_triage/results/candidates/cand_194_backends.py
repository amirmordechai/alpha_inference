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
        "plain_title": "Moderate Micro-Batching & V2 Runner Enablement",
        "hypothesis": (
            "The previous max-num-seqs=8 was too restrictive and hurt throughput. Increasing max-num-seqs to 20 "
            "and max-num-batched-tokens to 8192 allows for moderate micro-batching. At the same time, we enable "
            "the V2 model runner and set OMP_NUM_THREADS=4 to reduce host-side overhead."
        ),
        "category": "backends",  # vllm_flags | cuda_graphs | backends | triton_kernels
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
            "--max-num-seqs=20",
            "--max-num-batched-tokens=8192",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--disable-uvicorn-access-log",
            '--compilation-config={"cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]}',
        ],
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "custom_kernel_code": """try:
    import torch
    import torch._inductor.config as config
    config.coordinate_descent_tuning = False
    config.coordinate_descent_check_all_directions = False
    config.triton.cooperative_reduction = True
    config.triton.unique_kernel_names = True
    config.force_fuse_int_mm_with_mul = True
    config.fx_graph_cache = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
