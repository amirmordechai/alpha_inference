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
        "plain_title": "Drop low-accept spec decode + Triton router norm",
        "hypothesis": (
            "Eliminating ngram_gpu speculation (which had 12.6% acceptance while 4x-ing decode batch "
            "tokens in fused_moe) and fusing Gemma4 router RMSNorm+scale into a single-pass Triton kernel "
            "significantly reduces per-step ITL and P99 decode latency."
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
            "--max-num-seqs=256",
            "--max-num-batched-tokens=8192",
            "--quantization=fp8",
            "--language-model-only",
            "--async-scheduling",
            "--long-prefill-token-threshold=2048",
            '--kernel-config={"moe_backend":"triton","enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":512,"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,320,384,448,512]}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            **dict.fromkeys(
                (1, 2, 4, 8, 16, 32, 64),
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 4,
                },
            ),
            **dict.fromkeys(
                (128, 256),
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                    "num_warps": 4,
                    "num_stages": 4,
                },
            ),
            512: {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
                "num_stages": 4,
            },
            1024: {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_warps": 8,
                "num_stages": 4,
            },
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl
from sitecustomize import register_vllm_patch


@triton.jit
def _gemma4_router_rmsnorm_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    stride_x,
    stride_out,
    N: tl.constexpr,
    eps: tl.constexpr,
    root_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps) * root_size
    out = (x * rstd * s).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_out + cols, out, mask=mask)


def _patch_gemma4_router(mod):
    for cls_name in dir(mod):
        cls = getattr(mod, cls_name, None)
        if not isinstance(cls, type):
            continue
        if "Router" in cls_name and hasattr(cls, "forward"):
            orig_forward = cls.forward

            def _make_fast_forward(orig_fn):
                def _fast_forward(self, x: torch.Tensor):
                    if (
                        not x.is_cuda
                        or not hasattr(self, "scale")
                        or not hasattr(self, "root_size")
                        or not hasattr(self, "proj")
                        or not hasattr(self, "norm")
                    ):
                        return orig_fn(self, x)
                    orig_shape = x.shape
                    N = orig_shape[-1]
                    BLOCK_N = triton.next_power_of_2(N)
                    if BLOCK_N > 8192:
                        return orig_fn(self, x)
                    x_2d = x.reshape(-1, N)
                    M = x_2d.shape[0]
                    if M == 0:
                        return orig_fn(self, x)
                    out = torch.empty_like(x_2d)
                    eps = float(getattr(self.norm, "variance_epsilon", 1e-6))
                    root_size = float(self.root_size)
                    num_warps = 4 if BLOCK_N <= 2048 else 8
                    _gemma4_router_rmsnorm_scale_kernel[(M,)](
                        x_2d,
                        self.scale,
                        out,
                        x_2d.stride(0),
                        out.stride(0),
                        N=N,
                        eps=eps,
                        root_size=root_size,
                        BLOCK_N=BLOCK_N,
                        num_warps=num_warps,
                    )
                    out = out.view(orig_shape)
                    router_logits, _ = self.proj(out)
                    return router_logits

                return _fast_forward

            cls.forward = _make_fast_forward(orig_forward)


register_vllm_patch("vllm.model_executor.models.gemma4", _patch_gemma4_router)
register_vllm_patch("vllm.model_executor.models.gemma4_moe", _patch_gemma4_router)
""",
    }
# EVOLVE-BLOCK-END
