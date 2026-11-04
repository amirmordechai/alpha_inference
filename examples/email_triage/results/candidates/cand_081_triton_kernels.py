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
        "plain_title": "Dense CUDA Graphs + Micro-Batching (8/4096) + Fused Triton Softcap",
        "hypothesis": (
            "Drop max-num-seqs to 8 and max-num-batched-tokens to 4096 to prevent head-of-line blocking, "
            "add a dense CUDA graph capture ladder up to 4096, and inject a custom Triton kernel for fused "
            "logit softcapping to eliminate memory-bandwidth bottlenecks."
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
            "--max-num-batched-tokens=4096",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--disable-uvicorn-access-log",
            '--compilation-config={"cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096], "max_cudagraph_capture_size": 4096}',
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "custom_kernel_code": """
import gc
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_softcap_kernel(
    logits_ptr,
    soft_cap,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(logits_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    capped = tl.math.tanh(x_f32 / soft_cap) * soft_cap
    tl.store(logits_ptr + offsets, capped.to(x.dtype), mask=mask)

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
            
            # Apply our heavily optimized fused Triton kernel inplace
            if orig_soft_cap is not None:
                logits = result[0] if isinstance(result, tuple) else result
                if isinstance(logits, torch.Tensor):
                    n_elements = logits.numel()
                    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                    _fused_softcap_kernel[grid](logits, float(orig_soft_cap), n_elements, BLOCK_SIZE=1024)
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
