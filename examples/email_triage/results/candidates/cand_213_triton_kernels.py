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
        "plain_title": "Mega CUDA Graphs & Fused Triton Softcapping",
        "hypothesis": (
            "Crazy Idea: Host overhead for large prefills is silently ruining P99. "
            "We massively scale max_cudagraph_capture_size to 3072 with a hyper-dense ladder "
            "to eliminate CPU launch overhead for almost all prompts. To prevent mega-batch "
            "HoL blocking, we strictly clamp max-num-seqs to 8 and max-num-batched-tokens to 3072. "
            "We also inject a fused Triton kernel to do inplace logit softcapping over the 262k vocab."
        ),
        "category": "triton_kernels",  # vllm_flags | cuda_graphs | backends | triton_kernels
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
            "--max-num-seqs=8",
            "--max-num-batched-tokens=3072",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            '--compilation-config={"cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072], "max_cudagraph_capture_size": 3072}',
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
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
    x_f32 = x.to(tl.float32)
    x_f32 = x_f32 / cap
    x_f32 = tl.math.tanh(x_f32)
    y = x_f32 * cap
    tl.store(logits_ptr + offsets, y.to(x.dtype), mask=mask)

try:
    import vllm.model_executor.layers.logits_processor as lp
    orig_forward = lp.LogitsProcessor.forward

    def patched_forward(self, *args, **kwargs):
        # Steal the softcap so the original PyTorch forward skips it
        cap = getattr(self, "soft_cap", None)
        if cap is not None:
            self.soft_cap = None
            
        logits = orig_forward(self, *args, **kwargs)
        
        # Inject our blazing fast fused kernel in-place
        if cap is not None:
            self.soft_cap = cap
            if isinstance(logits, torch.Tensor) and logits.is_cuda:
                n_elements = logits.numel()
                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                fused_softcap_kernel[grid](logits, float(cap), n_elements, BLOCK_SIZE=1024)
                
        return logits

    lp.LogitsProcessor.forward = patched_forward
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
