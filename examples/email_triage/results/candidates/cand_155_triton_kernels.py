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
        "plain_title": "Custom Triton Softcap + Tuned Batching (16/4096)",
        "hypothesis": (
            "Replace slow PyTorch logit softcapping over 262k vocab with a custom Triton kernel. "
            "Increase micro-batching to 16/4096 to absorb burst requests while avoiding head-of-line blocking."
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
            "--max-num-seqs=16",
            "--max-num-batched-tokens=4096",
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
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "custom_kernel_code": """
import gc
import torch
import triton
import triton.language as tl

@triton.jit
def softcap_kernel(
    logits_ptr,
    soft_cap,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    logits = tl.load(logits_ptr + offsets, mask=mask)
    logits_f32 = logits.to(tl.float32)
    
    # Triton math handles the tanh directly
    logits_f32 = tl.math.tanh(logits_f32 / soft_cap) * soft_cap
    
    tl.store(logits_ptr + offsets, logits_f32.to(logits.dtype), mask=mask)

try:
    gc.collect()
    gc.freeze()

    import vllm.model_executor.layers.logits_processor as lp
    orig_forward = lp.LogitsProcessor.forward

    def fast_forward(self, *args, **kwargs):
        orig_soft_cap = getattr(self, 'soft_cap', None)
        try:
            if orig_soft_cap is not None:
                self.soft_cap = None
            
            result = orig_forward(self, *args, **kwargs)
            
            if orig_soft_cap is not None:
                logits = result[0] if isinstance(result, tuple) else result
                if isinstance(logits, torch.Tensor):
                    try:
                        n_elements = logits.numel()
                        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                        softcap_kernel[grid](logits, orig_soft_cap, n_elements, BLOCK_SIZE=1024)
                    except Exception:
                        # Safe fallback to eager FP32 math
                        logits_f32 = logits.to(torch.float32)
                        logits_f32.div_(orig_soft_cap).tanh_().mul_(orig_soft_cap)
                        logits.copy_(logits_f32)
            return result
        finally:
            if orig_soft_cap is not None:
                self.soft_cap = orig_soft_cap

    lp.LogitsProcessor.forward = fast_forward
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
