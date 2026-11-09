"""Seed inference configuration evolved by Google Cloud AlphaEvolve (`candidate.py`).

AlphaEvolve mutates the function inside the evolve block below to discover faster
vLLM scheduler flags, CUDA graph capture ladders, runtime backends, and custom
Triton GPU kernels tailored to your workload trace and GPU hardware.
"""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
# IMPORTANT: Keep the function name `get_candidate_config`.
# Supported dict keys: plain_title, hypothesis, category, vllm_args, env_vars, moe_tuned_config, custom_kernel_code.
def get_candidate_config() -> dict[str, Any]:
    """Return the vLLM serving configuration, optional MoE tile schedule, and optional Triton GPU kernel."""
    return {
        "plain_title": "1-tok high-precision ngram + Triton router + L2 MoE",
        "hypothesis": (
            "3-token ngram inflated MoE batch 4x on low-confidence matches. Capping ngram_gpu "
            "at 1 token with prompt_lookup_min=3 captures high-certainty JSON/template hits "
            "with minimal batch bloat, paired with Blackwell L2-swizzled MoE tiles and a fused Triton router norm."
        ),
        "category": "triton_kernels",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=16",
            "--gpu-memory-utilization=0.94",
            "--quantization=fp8",
            "--async-scheduling",
            "--language-model-only",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=1024",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":1,"prompt_lookup_max":5,"prompt_lookup_min":3}',
            '--kernel-config={"enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":512,"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512]}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            m: {
                "BLOCK_SIZE_M": 16 if m <= 128 else 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 16,
                "num_warps": 4,
                "num_stages": 5 if m <= 128 else 4,
            }
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl
from sitecustomize import register_vllm_patch

@triton.jit
def _fused_gemma4_router_norm_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    stride_xm,
    N: tl.constexpr,
    eps: tl.constexpr,
    root_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    y = x * (rstd * root_size) * s
    tl.store(out_ptr + row * stride_xm + cols, y.to(x_ptr.dtype.element_ty), mask=mask)

def _patch_gemma4(mod):
    RouterCls = getattr(mod, "Gemma4Router", None)
    if RouterCls is None or getattr(RouterCls, "_triton_patched", False):
        return

    orig_forward = RouterCls.forward

    def fast_router_forward(self, x: torch.Tensor):
        if not x.is_cuda or not x.is_contiguous():
            return orig_forward(self, x)
        N = x.shape[-1]
        BLOCK_N = triton.next_power_of_2(N)
        if BLOCK_N > 8192:
            return orig_forward(self, x)
        M = x.numel() // N
        out = torch.empty_like(x)
        _fused_gemma4_router_norm_scale_kernel[(M,)](
            x,
            self.scale,
            out,
            N,
            N=N,
            eps=float(self.norm.variance_epsilon),
            root_size=float(self.root_size),
            BLOCK_N=BLOCK_N,
            num_warps=4 if BLOCK_N <= 2048 else 8,
        )
        logits, _ = self.proj(out)
        return self.topk(logits)

    RouterCls.forward = fast_router_forward
    RouterCls._triton_patched = True

register_vllm_patch("vllm.model_executor.models.gemma4", _patch_gemma4)
""",
    }
# EVOLVE-BLOCK-END
