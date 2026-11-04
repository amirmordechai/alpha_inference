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
        "plain_title": "Tight Micro-Batching & Custom Triton Softcap",
        "hypothesis": (
            "Crazy Idea: Drastically tighten micro-batching (seqs=8, tokens=3072) to eliminate "
            "head-of-line blocking on single-step diffusion, while keeping dense CUDA graphs (max 2048). "
            "Also inject a custom Triton kernel monkey-patching vLLM's LogitsProcessor to fuse "
            "the memory-bound tanh logit softcapping over the 256k vocabulary."
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
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--disable-uvicorn-access-log",
            '--compilation-config={"cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048], "max_cudagraph_capture_size": 2048}',
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_tanh_softcap_kernel(
    logits_ptr,
    cap_val: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(logits_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    x_scaled = x_fp32 / cap_val
    x_tanh = tl.math.tanh(x_scaled)
    x_out = x_tanh * cap_val
    
    tl.store(logits_ptr + offsets, x_out.to(x.dtype), mask=mask)

try:
    import vllm.model_executor.layers.logits_processor as lp
    
    _orig_forward = lp.LogitsProcessor.forward
    
    def _fast_forward(self, *args, **kwargs):
        orig_soft_cap = getattr(self, "soft_cap", None)
        if orig_soft_cap is None:
            orig_soft_cap = getattr(self, "logits_soft_cap", None)
            
        if orig_soft_cap is not None:
            if hasattr(self, "soft_cap"):
                self.soft_cap = None
            if hasattr(self, "logits_soft_cap"):
                self.logits_soft_cap = None
                
        logits = _orig_forward(self, *args, **kwargs)
        
        if orig_soft_cap is not None:
            if hasattr(self, "soft_cap"):
                self.soft_cap = orig_soft_cap
            if hasattr(self, "logits_soft_cap"):
                self.logits_soft_cap = orig_soft_cap
                
            n_elements = logits.numel()
            BLOCK_SIZE = 4096
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            cap_val = float(orig_soft_cap)
            _fused_tanh_softcap_kernel[grid](
                logits,
                cap_val,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
        return logits

    lp.LogitsProcessor.forward = _fast_forward
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
