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
        "plain_title": "Fused Router Triton + Precision N-Gram + 768 Graphs",
        "hypothesis": (
            "Filtering ngram_gpu to min=3/k=2 eliminates low-probability draft waste on fused_moe, "
            "extending CUDA graphs to 768 covers full speculative batches, and fusing Gemma4Router RMSNorm+scale in Triton cuts launch overhead."
        ),
        "category": "triton_kernels",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=16",
            "--gpu-memory-utilization=0.95",
            "--quantization=fp8",
            "--kv-cache-dtype=fp8",
            "--language-model-only",
            "--async-scheduling",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=4096",
            "--long-prefill-token-threshold=1024",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":2,"prompt_lookup_max":6,"prompt_lookup_min":3}',
            '--compilation-config={"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,544,576,608,640,672,704,736,768],"max_cudagraph_capture_size":768}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            1: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            2: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            4: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            8: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            32: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            64: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            128: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            256: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            512: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            1024: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4},
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl
from sitecustomize import register_vllm_patch

@triton.jit
def _fused_gemma4_router_norm_scale_kernel(
    X_ptr,
    Scale_ptr,
    Out_ptr,
    stride_xm,
    N: tl.constexpr,
    root_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(Scale_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) * (1.0 / N)
    rstd = tl.rsqrt(var + eps) * root_size
    out = (x * rstd) * s
    tl.store(Out_ptr + row * stride_xm + cols, out.to(X_ptr.dtype.element_ty), mask=mask)

def _patch_gemma4_router(mod):
    RouterCls = getattr(mod, "Gemma4Router", None)
    if RouterCls is None or getattr(RouterCls, "_triton_patched", False):
        return
    orig_forward = RouterCls.forward

    def fast_router_forward(self, x: torch.Tensor):
        if not x.is_cuda or not x.is_contiguous():
            return orig_forward(self, x)
        orig_shape = x.shape
        N = orig_shape[-1]
        M = x.numel() // N
        BLOCK_N = triton.next_power_of_2(N)
        if BLOCK_N > 8192:
            return orig_forward(self, x)
        out = torch.empty_like(x)
        _fused_gemma4_router_norm_scale_kernel[(M,)](
            x,
            self.scale,
            out,
            N,
            N=N,
            root_size=float(self.root_size),
            eps=float(self.eps),
            BLOCK_N=BLOCK_N,
            num_warps=4 if BLOCK_N <= 2048 else 8,
        )
        router_logits, _ = self.proj(out)
        topk_weights, topk_ids = self.topk(router_logits)
        topk_weights = topk_weights * self.per_expert_scale.index_select(0, topk_ids.view(-1)).view_as(topk_weights)
        return topk_weights, topk_ids

    RouterCls.forward = fast_router_forward
    RouterCls._triton_patched = True

register_vllm_patch("vllm.model_executor.models.gemma4", _patch_gemma4_router)
""",
    }
# EVOLVE-BLOCK-END
