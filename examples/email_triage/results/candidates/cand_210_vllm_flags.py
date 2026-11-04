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
        "plain_title": "Triton Softcap Fusion & Tensor-Decomposed Micro-Batching",
        "hypothesis": (
            "Inspired by tensor decomposition, we 'decompose' large heavy bursts into tighter "
            "micro-batches (--max-num-seqs=10, --max-num-batched-tokens=4096) to reduce head-of-line "
            "blocking. We also increase the Triton kernel BLOCK_SIZE to 4096 and num_warps to 8 to better "
            "saturate Blackwell SMs and optimize memory bandwidth."
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
            "--max-num-seqs=10",
            "--max-num-batched-tokens=4096",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.92",
            "--async-scheduling",
            "--enable-prefix-caching",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl

@triton.jit
def fused_softcap_kernel(logits_ptr, cap, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(logits_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32) / cap
    x_f32 = tl.math.tanh(x_f32) * cap
    tl.store(logits_ptr + offsets, x_f32.to(x.dtype), mask=mask)

try:
    import vllm.model_executor.layers.logits_processor as lp
    if not hasattr(lp.LogitsProcessor, '_original_forward_patched'):
        lp.LogitsProcessor._original_forward_patched = lp.LogitsProcessor.forward
        
        def fast_forward(self, *args, **kwargs):
            # Temporarily disable native softcapping to avoid double ops
            cap = getattr(self, 'soft_cap', None)
            if cap is not None:
                self.soft_cap = None
            
            logits = self._original_forward_patched(*args, **kwargs)
            
            # Apply our fused triton kernel strictly inplace
            if cap is not None:
                self.soft_cap = cap
                if not logits.is_contiguous():
                    logits = logits.contiguous()
                n_elements = logits.numel()
                BLOCK_SIZE = 4096
                grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
                fused_softcap_kernel[grid](logits, cap, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
                
            return logits
            
        lp.LogitsProcessor.forward = fast_forward
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
