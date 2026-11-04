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
        "plain_title": "Balanced Micro-Batching & Advanced Inductor Optimizations",
        "hypothesis": (
            "Optimize micro-batching with max-num-seqs=16 and max-num-batched-tokens=8192 to "
            "leverage Blackwell's immense compute while avoiding head-of-line blocking. Disable "
            "large custom CUDA graph captures to prevent startup timeouts, and enable shape padding "
            "and loop order optimizations via Inductor configs."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
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
            "--max-num-seqs=16",
            "--max-num-batched-tokens=8192",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.90",
            "--disable-uvicorn-access-log",
            "--disable-log-stats",
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
        },
        "custom_kernel_code": """
import torch
try:
    import torch._inductor.config as config
    config.fx_graph_cache = True
    config.compile_threads = 8
    config.triton.unique_kernel_names = True
    config.shape_padding = True
    config.size_asserts = False
    config.pick_loop_orders = True
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
