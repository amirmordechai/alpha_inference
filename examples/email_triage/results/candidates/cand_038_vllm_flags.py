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
        "plain_title": "Crazy Simplicity: No Caching, No Async, Micro-Batches",
        "hypothesis": (
            "Because this is a 1-step diffusion model, autoregressive decoding optimizations "
            "like prefix caching and async scheduling are pure overhead. The Crazy Idea: "
            "completely strip them out to simplify the codebase and eliminate block matching/sync "
            "latencies. We also brutally restrict max-num-seqs to 8 and max-num-batched-tokens "
            "to 2048 to force lightning-fast, uninterrupted single-step forward passes."
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
            '--limit-mm-per-prompt={"image": 0, "video": 0}',
            "--attention-backend=TRITON_ATTN",
            "--max-num-seqs=8",
            "--max-num-batched-tokens=2048",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "true",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
