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
        "plain_title": "Ultra-Wide CUDA Graphs & Micro-Batch Strangulation",
        "hypothesis": (
            "Push max_cudagraph_capture_size to 2048 to entirely envelop 1-step diffusion "
            "passes in zero-overhead FULL CUDA graphs, paired with strict micro-batch limits "
            "(max-num-seqs=8, max-num-batched-tokens=3072) to eliminate head-of-line blocking."
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
            "--max-num-seqs=8",
            "--max-num-batched-tokens=3072",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--disable-uvicorn-access-log",
            '--compilation-config={"max_cudagraph_capture_size": 2048, "pass_config": {}}',
            "--long-prefill-token-threshold=8192",
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
        },
        "custom_kernel_code": """
try:
    import vllm.worker.model_runner
    ladder = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    if hasattr(vllm.worker.model_runner, "_BATCH_SIZES_TO_CAPTURE"):
        vllm.worker.model_runner._BATCH_SIZES_TO_CAPTURE = ladder
    if hasattr(vllm.worker.model_runner, "_TOKEN_BUDGETS_TO_CAPTURE"):
        vllm.worker.model_runner._TOKEN_BUDGETS_TO_CAPTURE = ladder
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
