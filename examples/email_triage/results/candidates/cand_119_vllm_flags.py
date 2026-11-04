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
        "plain_title": "Moderate micro-batches to balance queuing and HOL blocking",
        "hypothesis": (
            "Increasing --max-num-seqs to 24 and --max-num-batched-tokens to 8192 strikes the "
            "optimal balance between absorbing bursty requests to minimize queuing delay and "
            "preventing head-of-line blocking. Setting --gpu-memory-utilization to 0.95 provides "
            "ample headroom for batching and CUDA graphs."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
            "--served-model-name=diffusiongemma-26b",
            "--host=0.0.0.0",
            "--port=8080",
            '--diffusion-config={"canvas_length": 64}',
            "--max-logprobs=5",
            "--disable-uvicorn-access-log",
            '--override-generation-config={"max_new_tokens": null}',
            "--enable-auto-tool-choice",
            "--tool-call-parser=gemma4",
            "--reasoning-parser=gemma4",
            '--limit-mm-per-prompt={"image": 0, "video": 0}',
            "--attention-backend=TRITON_ATTN",
            "--max-num-seqs=24",
            "--max-num-batched-tokens=8192",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--async-scheduling",
            "--enable-prefix-caching",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
