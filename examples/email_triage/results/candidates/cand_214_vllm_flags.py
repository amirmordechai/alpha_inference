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
        "plain_title": "V1 Engine with Balanced Micro-Batching & Expanded CUDA Graphs",
        "hypothesis": (
            "We transition back to the mature V1 engine (VLLM_USE_V2_MODEL_RUNNER=0) and remove the "
            "custom Triton softcap patch to eliminate graph breaks and let the highly optimized native "
            "vLLM paths run natively. We balance micro-batching (--max-num-seqs=16, --max-num-batched-tokens=8192) "
            "to absorb bursts without splitting single-step diffusion requests, and expand CUDA graph "
            "captures up to 1024 tokens to maximize GPU utilization on Blackwell."
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
            "--attention-backend=TRITON_ATTN",
            "--max-num-seqs=16",
            "--max-num-batched-tokens=8192",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--max-seq-len-to-capture=1024",
            "--async-scheduling",
            "--enable-prefix-caching",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
        },
        "custom_kernel_code": "# No custom kernel needed; relying on vLLM V1 highly optimized native paths.",
    }
# EVOLVE-BLOCK-END
