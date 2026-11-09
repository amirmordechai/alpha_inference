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
        "plain_title": "On-Chip Register-Fused Router RMSNorm+GEMM",
        "hypothesis": (
            "Eliminate both the intermediate HBM round-trip and the 1260 small [M, 2816]x[2816, 128] "
            "cuBLAS router projection GEMMs by fusing RMSNorm + root_size + scale + Linear(E=128) "
            "into a single L2/register-resident Triton kernel alongside dense CUDA graph capture."
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
            "--language-model-only",
            "--async-scheduling",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=2048",
            "--disable-uvicorn-access-log",
            '--compilation-config={"max_cudagraph_capture_size":512}',
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


@triton.jit
def _fused_router_norm_scale_gemm_kernel(
    x_ptr,
    scale_ptr,
    w_ptr,
    logits_ptr,
    M,
    stride_xm,
    stride_we,
    stride_lm,
    K: tl.constexpr,
    E: tl.constexpr,
    eps: tl.constexpr,
    root_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Compute exact row-wise rrms across full K in float32
    cols_full = tl.arange(0, FULL_K)
    mask_full = cols_full < K
    x_full = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + cols_full[None, :],
        mask=mask_m[:, None] & mask_full[None, :],
        other=0.0,
    ).to(tl.float32)
    var = tl.sum(x_full * x_full, axis=1) / K
    rrms = tl.rsqrt(var + eps) * root_size

    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        s_tile = tl.load(scale_ptr + offs_k, mask=mask_k, other=1.0).to(tl.float32)
        x_norm = ((x_tile * rrms[:, None]) * s_tile[None, :]).to(w_ptr.dtype.element_ty)

        w_tile = tl.load(
            w_ptr + offs_e[None, :] * stride_we + offs_k[:, None],
            mask=mask_k[:, None] & mask_e[None, :],
            other=0.0,
        )
        acc = tl.dot(x_norm, w_tile, acc)

    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_e[None, :],
        acc.to(logits_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_e[None, :],
    )


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
        eps = float(getattr(self.norm, "variance_epsilon", 1e-6))
        root_size = float(self.root_size)

        w = getattr(self.proj, "weight", None)
        if (
            w is not None
            and w.dtype == x.dtype
            and w.is_contiguous()
            and getattr(self.proj, "bias", None) is None
            and M <= 512
            and w.shape[0] == 128
        ):
            E = w.shape[0]
            logits = torch.empty((M, E), dtype=x.dtype, device=x.device)
            BLOCK_M = 16 if M <= 32 else 32
            BLOCK_E = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(E, BLOCK_E))
            _fused_router_norm_scale_gemm_kernel[grid](
                x_2d,
                self.scale,
                w,
                logits,
                M,
                x_2d.stride(0),
                w.stride(0),
                logits.stride(0),
                K=N,
                E=E,
                eps=eps,
                root_size=root_size,
                BLOCK_M=BLOCK_M,
                BLOCK_E=BLOCK_E,
                BLOCK_K=128,
                FULL_K=BLOCK_N,
                num_warps=4,
                num_stages=3,
            )
            return self.topk(logits.view(*orig_shape[:-1], E))

        out = torch.empty_like(x_2d)
        _fused_router_norm_scale_kernel[(M,)](
            x_2d,
            self.scale,
            out,
            x_2d.stride(0),
            N=N,
            eps=eps,
            root_size=root_size,
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
