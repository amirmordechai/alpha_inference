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
        "plain_title": "Monolithic Router GEMM Triton Kernel + NoSpec",
        "hypothesis": (
            "Disabling low-acceptance (16.9%) ngram speculation eliminates 3x MoE token amplification, while "
            "fusing the entire Gemma-4 Router RMSNorm + Scale + 128-Expert Linear Projection into a single "
            "in-SRAM Triton kernel eliminates intermediate global memory roundtrips and tiny GEMM launches."
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
            '--kernel-config={"enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":256,"cudagraph_capture_sizes":[1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            m: {
                "BLOCK_SIZE_M": 16 if m <= 128 else 32,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1 if m <= 32 else 8,
                "num_warps": 4,
                "num_stages": 5,
            }
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl
from sitecustomize import register_vllm_patch


@triton.jit
def _fused_router_norm_scale_proj_kernel(
    x_ptr,
    scale_ptr,
    w_ptr,
    logits_ptr,
    M,
    stride_xm,
    stride_we,
    stride_lm,
    N: tl.constexpr,
    E: tl.constexpr,
    eps: tl.constexpr,
    root_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Pass 1: Compute exact fp32 variance across N for BLOCK_M rows
    var = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, N, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < N
        x_chunk = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        var += tl.sum(x_chunk * x_chunk, axis=1)

    rstd = tl.rsqrt(var / N + eps) * root_size

    # Pass 2: Normalize, scale, and accumulate TensorCore dot product with W^T [N, E]
    offs_e = tl.arange(0, E)
    acc = tl.zeros((BLOCK_M, E), dtype=tl.float32)
    for k0 in range(0, N, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < N
        x_chunk = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        s_chunk = tl.load(scale_ptr + offs_k, mask=mask_k, other=1.0).to(tl.float32)
        h_chunk = (x_chunk * rstd[:, None] * s_chunk[None, :]).to(w_ptr.dtype.element_ty)

        # Load w_chunk [BLOCK_K, E] from W [E, N] via transposed pointers
        w_chunk = tl.load(
            w_ptr + offs_e[None, :] * stride_we + offs_k[:, None],
            mask=mask_k[:, None],
            other=0.0,
        )
        acc = tl.dot(h_chunk, w_chunk, acc)

    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_e[None, :],
        acc.to(w_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


def _fused_router_norm_scale_proj(
    x: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor, eps: float, root_size: float
) -> torch.Tensor:
    N = x.shape[-1]
    E = weight.shape[0]
    x_2d = x.contiguous().view(-1, N)
    M = x_2d.shape[0]
    logits = torch.empty((M, E), dtype=weight.dtype, device=x.device)
    BLOCK_M = 16 if M <= 64 else 32
    BLOCK_K = 128
    grid = (triton.cdiv(M, BLOCK_M),)
    _fused_router_norm_scale_proj_kernel[grid](
        x_2d,
        scale,
        weight,
        logits,
        M,
        x_2d.stride(0),
        weight.stride(0),
        logits.stride(0),
        N=N,
        E=E,
        eps=eps,
        root_size=root_size,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return logits


def _patch_gemma4(mod):
    for attr_name in dir(mod):
        cls = getattr(mod, attr_name, None)
        if not isinstance(cls, type):
            continue
        if "Router" in attr_name and hasattr(cls, "forward"):
            orig_forward = cls.forward

            def _make_router_forward(orig_fn):
                def _router_forward(self, x):
                    if (
                        hasattr(self, "norm")
                        and hasattr(self, "scale")
                        and hasattr(self, "_root_size")
                        and hasattr(self, "proj")
                        and hasattr(self, "topk")
                        and hasattr(self.proj, "weight")
                        and self.proj.weight.dtype in (torch.bfloat16, torch.float16)
                        and self.proj.weight.shape[0] == 128
                    ):
                        eps = getattr(self.norm, "variance_epsilon", 1e-6)
                        router_logits = _fused_router_norm_scale_proj(
                            x, self.scale, self.proj.weight, float(eps), float(self._root_size)
                        )
                        return self.topk(router_logits)
                    return orig_fn(self, x)

                return _router_forward

            cls.forward = _make_router_forward(orig_forward)


register_vllm_patch("vllm.model_executor.models.gemma4", _patch_gemma4)
""",
    }
# EVOLVE-BLOCK-END
