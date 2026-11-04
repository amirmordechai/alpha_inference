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
        "plain_title": "Inplace PyTorch Softcap + Moderate Batching (12/3072) + Optimised CUDA Graphs",
        "hypothesis": (
            "Replacing slow out-of-place softcapping with 100% exact-match inplace PyTorch operations to eliminate memory allocations. "
            "Tuning micro-batching to 12/3072 and reducing CUDA graph capture max size to 2048 to minimize warmup overhead and latency."
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
            "--max-num-seqs=12",
            "--max-num-batched-tokens=3072",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.90",
            "--disable-uvicorn-access-log",
            '--compilation-config={"cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048], "max_cudagraph_capture_size": 2048}',
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "custom_kernel_code": """
import gc
import torch

try:
    gc.collect()
    gc.freeze()

    import vllm.model_executor.layers.logits_processor as lp
    orig_forward = lp.LogitsProcessor.forward

    def fast_forward(self, *args, **kwargs):
        orig_soft_cap = getattr(self, 'soft_cap', None)
        try:
            # Temporarily disable standard softcapping so eager PyTorch math isn't fired
            if orig_soft_cap is not None:
                self.soft_cap = None
            
            result = orig_forward(self, *args, **kwargs)
            
            # Apply our highly optimized inplace softcap
            if orig_soft_cap is not None:
                logits = result[0] if isinstance(result, tuple) else result
                if isinstance(logits, torch.Tensor):
                    logits.div_(orig_soft_cap).tanh_().mul_(orig_soft_cap)
            return result
        finally:
            # Safely restore bounds for future checks/calls
            if orig_soft_cap is not None:
                self.soft_cap = orig_soft_cap

    lp.LogitsProcessor.forward = fast_forward
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
