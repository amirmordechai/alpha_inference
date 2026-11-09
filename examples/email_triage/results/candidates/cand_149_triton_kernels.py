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
    custom_kernel_code = """
import torch
import triton
import triton.language as tl
from sitecustomize import register_vllm_patch

@triton.jit
def _fused_router_norm_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    stride_x,
    N,
    root_size,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    y = x * (rstd * root_size) * s
    tl.store(out_ptr + row * stride_x + cols, y.to(x_ptr.dtype.element_ty), mask=mask)

def _patch_gemma4(mod):
    router_cls = getattr(mod, "Gemma4Router", None) or getattr(mod, "Gemma4MoERouter", None)
    if router_cls is None:
        return
    orig_forward = router_cls.forward

    def _fast_router_forward(self, x: torch.Tensor):
        if (
            x.is_cuda
            and hasattr(self, "scale")
            and hasattr(self, "root_size")
            and hasattr(self, "norm")
            and hasattr(self, "proj")
        ):
            orig_shape = x.shape
            N = orig_shape[-1]
            x_2d = x.contiguous().view(-1, N)
            M = x_2d.shape[0]
            out = torch.empty_like(x_2d)
            eps = getattr(self.norm, "variance_epsilon", 1e-6)
            BLOCK_N = triton.next_power_of_2(N)
            _fused_router_norm_scale_kernel[(M,)](
                x_2d,
                self.scale,
                out,
                x_2d.stride(0),
                N,
                float(self.root_size),
                float(eps),
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )
            return self.proj(out.view(orig_shape))
        return orig_forward(self, x)

    router_cls.forward = _fast_router_forward

register_vllm_patch("vllm.model_executor.models.gemma4", _patch_gemma4)
"""

    return {
        "plain_title": "N=32 Sub-Tile MoE + No-Spec + Fused Router",
        "hypothesis": (
            "Eliminating 16.9%-accept ngram speculation removes 3x token amplification in fused_moe, "
            "while exact-divisor BLOCK_SIZE_N=32 (704=22*32) with 5 stages and a fused Triton router "
            "norm+scale kernel maximize SM120 occupancy and cut decode ITL."
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
            "--max-num-batched-tokens=4096",
            "--disable-uvicorn-access-log",
            '--kernel-config={"enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":512}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            1: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
            2: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
            4: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
            8: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
            16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
            32: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 5},
            64: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 5},
            128: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 5},
            256: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            512: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4},
            1024: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4},
        },
        "custom_kernel_code": custom_kernel_code,
    }
# EVOLVE-BLOCK-END
