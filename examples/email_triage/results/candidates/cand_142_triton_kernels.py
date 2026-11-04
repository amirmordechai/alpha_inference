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
        "plain_title": "Fused Triton Logit Softcapping + Denser CUDA Graph Ladder + Micro-batching tuning",
        "hypothesis": (
            "We implement a custom in-place Triton kernel for Gemma 2's final logit softcapping "
            "(tanh(x/cap)*cap) to bypass redundant memory allocations of the massive vocab tensor. "
            "We tune max-num-seqs to 12 and max-num-batched-tokens to 3072 to avoid head-of-line blocking, "
            "and define a dense cudagraph_sizes ladder up to 1024 to minimize padding overhead."
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
            "--max-num-seqs=12",
            "--max-num-batched-tokens=3072",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            '--compilation-config={"max_cudagraph_capture_size": 1024, "cudagraph_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024]}',
            "--disable-uvicorn-access-log",
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
    import torch.nn as nn
    import triton
    import triton.language as tl

    @triton.jit
    def fused_softcap_kernel(
        logits_ptr,
        cap,
        num_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        
        # Load logits
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0)
        
        # Perform softcapping in float32 for high precision
        x_f32 = x.to(tl.float32)
        scaled_x = x_f32 / cap
        tanh_x = tl.tanh(scaled_x)
        out_f32 = tanh_x * cap
        
        # Store back in-place (Triton cast back to original dtype)
        tl.store(logits_ptr + offsets, out_f32, mask=mask)

    # Lazy patch for Gemma 2 causal LM to intercept logits softcapping
    orig_module_init = nn.Module.__init__
    def patched_module_init(self, *args, **kwargs):
        orig_module_init(self, *args, **kwargs)
        if self.__class__.__name__ == "Gemma2ForCausalLM":
            try:
                cls = self.__class__
                if not getattr(cls, "_is_softcap_patched", False):
                    orig_compute_logits = getattr(cls, "compute_logits", None)
                    if orig_compute_logits is not None:
                        def patched_compute_logits(instance, hidden_states, lm_head):
                            logits = lm_head(hidden_states)
                            cap = getattr(instance.config, "final_logit_softcapping", None)
                            if cap is not None:
                                try:
                                    logits = logits.contiguous()
                                    num_elements = logits.numel()
                                    BLOCK_SIZE = 1024
                                    grid = ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
                                    fused_softcap_kernel[grid](
                                        logits,
                                        float(cap),
                                        num_elements,
                                        BLOCK_SIZE=BLOCK_SIZE,
                                    )
                                except Exception:
                                    # Safe fallback to standard PyTorch implementation
                                    logits = logits / cap
                                    logits = torch.tanh(logits)
                                    logits = logits * cap
                            return logits
                        cls.compute_logits = patched_compute_logits
                        cls._is_softcap_patched = True
            except Exception:
                pass
    nn.Module.__init__ = patched_module_init

    # Patch torch.tanh as secondary defense
    orig_tanh = torch.tanh
    def fast_tanh(input, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.is_cuda:
            if input.shape[-1] >= 250000 and not input.is_leaf and not input.requires_grad:
                try:
                    return input.tanh_()
                except Exception:
                    pass
        return orig_tanh(input, *args, **kwargs)
    torch.tanh = fast_tanh
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
