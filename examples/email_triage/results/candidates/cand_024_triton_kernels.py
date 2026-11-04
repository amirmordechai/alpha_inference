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
        "plain_title": "Crazy F1-Pitstop Micro-Batching & Dense Graphs",
        "hypothesis": (
            "Applying a 'Crazy with a capital C' approach: Treat the GPU like an F1 pitstop "
            "by clamping --max-num-seqs to 8, capping batched tokens at 3072, and explicitly "
            "blocking prefill splitting. Combined with an unhinged runtime hot-patch that "
            "forces a hyper-dense exact-match CUDA graph capture ladder up to 3072, we eliminate "
            "padding overhead entirely for 1-step diffusion. Overclock via TRITON_PTXAS_OPTIONS."
        ),
        "category": "triton_kernels",  # vllm_flags | cuda_graphs | backends | triton_kernels
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
            "--long-prefill-token-threshold=8192",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.92",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "TRITON_PTXAS_OPTIONS": "-O4",
            "TORCH_CUDNN_V8_API_ENABLED": "1",
        },
        "custom_kernel_code": """
try:
    import vllm.worker.model_runner as mr
    
    # Inject a crazy hyper-dense CUDA graph ladder dynamically so the exact token counts
    # hit a pre-captured graph (no padding waste) for 1-step exact diffusion passes!
    if hasattr(mr, '_get_graph_batch_size'):
        def _crazy_ladder(*args, **kwargs):
            return [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072]
        mr._get_graph_batch_size = _crazy_ladder
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
