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
        "plain_title": "FP8 + GPU N-Gram Speculation + Fused Router Norm",
        "hypothesis": (
            "Combine online FP8 quantization and tuned Triton MoE tiles with zero-VRAM GPU n-gram "
            "speculative decoding (ngram_gpu, 3 tokens) and a lightweight fused Router RMSNorm+scale "
            "Triton kernel to cut total decode steps and P99 latency."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=16",
            "--gpu-memory-utilization=0.94",
            "--quantization=fp8",
            "--language-model-only",
            "--async-scheduling",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=2048",
            "--disable-uvicorn-access-log",
            '--compilation-config={"max_cudagraph_capture_size":512}',
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            1: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            2: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            4: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            8: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            32: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            64: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            128: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            256: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4},
            512: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4},
            1024: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4},
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl
from sitecustomize import register_vllm_patch


@triton.jit
def _fused_router_norm_scale_kernel(
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
    rrms = tl.rsqrt(var + eps) * root_size
    out = (x * rrms) * s
    tl.store(out_ptr + row * stride_xm + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


def _patch_gemma4_router(mod):
    router_cls = getattr(mod, "Gemma4Router", None)
    if router_cls is None or getattr(router_cls, "_triton_patched", False):
        return

    orig_forward = router_cls.forward

    def _fast_router_forward(self, x: torch.Tensor):
        if not x.is_cuda or not x.is_contiguous():
            return orig_forward(self, x)
        orig_shape = x.shape
        N = orig_shape[-1]
        M = x.numel() // N
        BLOCK_N = triton.next_power_of_2(N)
        if BLOCK_N > 8192:
            return orig_forward(self, x)
        x_2d = x.view(M, N)
        out = torch.empty_like(x_2d)
        eps = float(getattr(self.norm, "variance_epsilon", 1e-6))
        _fused_router_norm_scale_kernel[(M,)](
            x_2d,
            self.scale,
            out,
            x_2d.stride(0),
            N=N,
            eps=eps,
            root_size=float(self.root_size),
            BLOCK_N=BLOCK_N,
            num_warps=4 if BLOCK_N <= 2048 else 8,
        )
        router_logits, _ = self.proj(out.view(orig_shape))
        return self.topk(router_logits)

    router_cls.forward = _fast_router_forward
    router_cls._triton_patched = True


register_vllm_patch("vllm.model_executor.models.gemma4", _patch_gemma4_router)
""",
    }
# EVOLVE-BLOCK-END
