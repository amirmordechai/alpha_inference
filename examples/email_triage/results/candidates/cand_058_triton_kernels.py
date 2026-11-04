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
        "plain_title": "Triton Softcap Fusion & V2 Engine for Extreme P99",
        "hypothesis": (
            "Dynamically inject a custom Triton kernel to fuse the memory-bound logit softcapping "
            "(262k vocab) into a single pass, saving ~6MB of HBM traffic per token. Combine this "
            "with the V2 model runner and tighter scheduler bounds (seqs=20, tokens=8192) to "
            "eliminate head-of-line blocking and minimize P99."
        ),
        "category": "triton_kernels",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
            "--served-model-name=diffusiongemma-26b",
            "--host=0.0.0.0",
            "--port=8080",
            '--diffusion-config={"canvas_length": 64}',
            "--max-logprobs=5",
            "--disable-uvicorn-access-log",
            '--override-generation-config={"max_new_tokens": null}',
            "--enable-auto-tool-choice",
            "--tool-call-parser=gemma4",
            "--reasoning-parser=gemma4",
            "--enable-prefix-caching",
            "--async-scheduling",
            '--limit-mm-per-prompt={"image": 0, "video": 0}',
            "--attention-backend=TRITON_ATTN",
            "--max-num-seqs=20",
            "--max-num-batched-tokens=8192",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        "custom_kernel_code": """
try:
    import gc
    gc.set_threshold(100000, 15, 15)
except Exception:
    pass

try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

try:
    import torch
    import triton
    import triton.language as tl
    import vllm.model_executor.layers.logits_processor as lp

    @triton.jit
    def _fused_softcap_kernel(
        logits_ptr,
        cap,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(logits_ptr + offsets, mask=mask)
        x_f32 = x.to(tl.float32)
        
        # Logit softcap formula: x = tanh(x / cap) * cap
        x_f32 = x_f32 / cap
        x_f32 = tl.math.tanh(x_f32)
        x_f32 = x_f32 * cap
        
        tl.store(logits_ptr + offsets, x_f32.to(x.dtype), mask=mask)

    _orig_forward = lp.LogitsProcessor.forward

    def _fast_softcap_forward(self, *args, **kwargs):
        # 1. Intercept and temporarily disable vLLM's slow native PyTorch softcapping
        caps = {}
        for attr in ['soft_cap', 'logit_softcapping', 'final_logit_softcapping']:
            val = getattr(self, attr, None)
            if val is not None:
                caps[attr] = val
                setattr(self, attr, None)
                
        # 2. Get the logits without softcapping applied
        logits = _orig_forward(self, *args, **kwargs)
        
        # 3. Apply our ultra-fast fused Triton kernel in-place
        if caps and logits is not None:
            # Restore attributes safely
            for attr, val in caps.items():
                setattr(self, attr, val)
                
            cap_val = list(caps.values())[0]
            if hasattr(cap_val, 'item'):
                cap_val = cap_val.item()
            cap_val = float(cap_val)
            
            n_elements = logits.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            _fused_softcap_kernel[grid](
                logits,
                cap_val,
                n_elements,
                BLOCK_SIZE=1024,
            )
            
        return logits

    # Inject the monkey-patch
    lp.LogitsProcessor.forward = _fast_softcap_forward

except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
