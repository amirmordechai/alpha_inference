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
        "plain_title": "Fused Triton Logit Softcapping & Balanced Micro-batching",
        "hypothesis": (
            "Implements a high-precision Triton kernel for final logit softcapping to avoid "
            "vocabulary tensor memory re-allocation overhead. Tunes max-num-seqs down to 12 and "
            "max-num-batched-tokens to 3072 to eliminate head-of-line blocking under bursty "
            "arrivals, while increasing max_cudagraph_capture_size to 1024 via compilation config."
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
            "--disable-uvicorn-access-log",
            '--compilation-config={"max_cudagraph_capture_size": 1024}',
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
        
        # Store back in-place (casting to original dtype)
        tl.store(logits_ptr + offsets, out_f32.to(x.dtype), mask=mask)

    # Class-level patch for final logit softcapping
    orig_module_init = nn.Module.__init__
    _patched_classes = set()

    def patched_module_init(self, *args, **kwargs):
        orig_module_init(self, *args, **kwargs)
        cls = self.__class__
        if cls not in _patched_classes:
            _patched_classes.add(cls)
            if hasattr(cls, "compute_logits"):
                try:
                    orig_compute_logits = getattr(cls, "compute_logits")
                    def patched_compute_logits(instance, hidden_states, lm_head, *args, **kwargs):
                        cap = getattr(instance.config, "final_logit_softcapping", None)
                        if cap is not None:
                            try:
                                logits = lm_head(hidden_states)
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
                                return logits
                            except Exception:
                                pass
                        return orig_compute_logits(instance, hidden_states, lm_head, *args, **kwargs)
                    
                    cls.compute_logits = patched_compute_logits
                except Exception:
                    pass

    nn.Module.__init__ = patched_module_init

except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
