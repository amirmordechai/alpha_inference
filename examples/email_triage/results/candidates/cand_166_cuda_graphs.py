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
        "plain_title": "Fix max-model-len and Evaluate Dense CUDA Graphs",
        "hypothesis": (
            "Fix the max-model-len error by setting it to 12288. Retain the dense CUDA "
            "graph capture sizes up to 512, micro-batching with max-num-seqs=12, and "
            "OMP_NUM_THREADS=8 to test if this sweet-spot improves P99 latency."
        ),
        "category": "cuda_graphs",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
            "--served-model-name=diffusiongemma-26b",
            "--host=0.0.0.0",
            "--port=8080",
            "--disable-uvicorn-access-log",
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
            "--max-num-seqs=12",
            "--max-num-batched-tokens=4096",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.90",
            '--compilation-config={"cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 320, 384, 448, 512], "max_cudagraph_capture_size": 512}',
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
