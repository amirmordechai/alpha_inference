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
        "plain_title": "Crazy Fused Softcap Triton Kernel & Tight Micro-Batching",
        "hypothesis": (
            "Drastically cut memory bandwidth bottleneck on 262k vocab logits softcapping "
            "by dynamically intercepting LogitsProcessor to run an in-place fused Triton tanh "
            "kernel, while severely tightening batch limits (seqs=12, tokens=4096) to prevent "
            "head-of-line blocking in the single-step diffusion forward pass."
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
            "--max-num-seqs=12",
            "--max-num-batched-tokens=4096",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.90",
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
        },
        "custom_kernel_code": """
try:
    import torch
    import triton
    import triton.language as tl
    import vllm.model_executor.layers.logits_processor as lp

    @triton.jit
    def fused_softcap_kernel(logits_ptr, softcap, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(logits_ptr + offsets, mask=mask)
        x_f32 = x.to(tl.float32)
        x_f32 = tl.math.tanh(x_f32 / softcap) * softcap
        tl.store(logits_ptr + offsets, x_f32.to(x.dtype), mask=mask)

    _orig_forward = lp.LogitsProcessor.forward

    def custom_forward(self, *args, **kwargs):
        soft_cap = getattr(self, "soft_cap", None)
        if soft_cap is not None:
            # Hide soft_cap to bypass vLLM's naive PyTorch implementation
            self.soft_cap = None
            try:
                logits = _orig_forward(self, *args, **kwargs)
            finally:
                self.soft_cap = soft_cap
                
            n_elements = logits.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            # Execute fused in-place operations across the massive vocab bounds
            fused_softcap_kernel[grid](logits, float(soft_cap), n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return logits
        else:
            return _orig_forward(self, *args, **kwargs)

    lp.LogitsProcessor.forward = custom_forward
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
