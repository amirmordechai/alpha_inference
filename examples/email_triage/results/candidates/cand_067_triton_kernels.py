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
        "plain_title": "Fused Tanh Softcap & Tight Micro-Batching",
        "hypothesis": (
            "Inject a custom Triton kernel to fuse the memory-bound tanh logit softcapping "
            "over 262k vocab logits, saving massive HBM bandwidth. Combine with a denser "
            "CUDA graph token ladder and tighter micro-batch limits (seqs=8, tokens=6144) "
            "to minimize head-of-line blocking."
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
            "--max-num-batched-tokens=6144",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "true",
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl

@triton.jit
def fused_softcap_kernel(logits_ptr, cap, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(logits_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    x_f32 = x_f32 / cap
    x_f32 = tl.math.tanh(x_f32)
    x_f32 = x_f32 * cap
    tl.store(logits_ptr + offsets, x_f32.to(x.dtype), mask=mask)

def apply_fused_softcap(logits, cap):
    if not logits.is_contiguous():
        logits = logits.contiguous()
    n_elements = logits.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_softcap_kernel[grid](logits, cap, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return logits

try:
    import vllm.model_executor.layers.logits_processor
    original_get_logits = vllm.model_executor.layers.logits_processor.LogitsProcessor._get_logits
    
    def custom_get_logits(self, hidden_states, lm_head, embedding_bias=None):
        cap = getattr(self, 'soft_cap', None)
        self.soft_cap = None
        logits = original_get_logits(self, hidden_states, lm_head, embedding_bias)
        self.soft_cap = cap
        
        if cap is not None:
            logits = apply_fused_softcap(logits, cap)
            
        return logits

    vllm.model_executor.layers.logits_processor.LogitsProcessor._get_logits = custom_get_logits
except Exception:
    pass

try:
    import vllm.worker.model_runner
    if hasattr(vllm.worker.model_runner, '_BATCH_SIZES_TO_CAPTURE'):
        vllm.worker.model_runner._BATCH_SIZES_TO_CAPTURE = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
