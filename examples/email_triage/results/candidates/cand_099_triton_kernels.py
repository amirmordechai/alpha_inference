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
        "plain_title": "Ultra-Sparse M=16 MoE Tiles + 2-Tok Spec + Router Fusion",
        "hypothesis": (
            "With E=128, top_k=8, even M=256 has only 16 tokens/expert on average; forcing "
            "BLOCK_SIZE_M=16 through M=256 and BLOCK_SIZE_M=32 at M=512..1024 while trimming "
            "ngram_gpu speculation from 3 to 2 tokens (since accept rate is 12.6%) and fusing "
            "Gemma4 Router RMSNorm+scale in Triton slashes wasted MoE FLOPs."
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
            "--long-prefill-token-threshold=1024",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":2,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            '--compilation-config={"max_cudagraph_capture_size":512}',
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
            16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
            32: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            64: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            128: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            256: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            512: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            1024: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4},
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
    rrms = tl.math.rsqrt(var + eps) * root_size
    y = (x * rrms) * s
    tl.store(out_ptr + row * stride_xm + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


def _patch_gemma4(mod):
    for attr_name in dir(mod):
        cls = getattr(mod, attr_name, None)
        if not isinstance(cls, type) or "Router" not in attr_name:
            continue
        orig_forward = getattr(cls, "forward", None)
        if orig_forward is None:
            continue

        def _make_forward(orig_fn):
            def _fast_forward(self, x):
                if (
                    hasattr(self, "norm")
                    and hasattr(self, "scale")
                    and hasattr(self, "root_size")
                    and hasattr(self, "proj")
                    and x.is_cuda
                    and x.is_contiguous()
                ):
                    orig_shape = x.shape
                    N = orig_shape[-1]
                    M = x.numel() // N
                    BLOCK_N = triton.next_power_of_2(N)
                    if BLOCK_N <= 8192:
                        out = torch.empty_like(x)
                        eps = getattr(self.norm, "variance_epsilon", 1e-6)
                        _fused_router_norm_scale_kernel[(M,)](
                            x,
                            self.scale,
                            out,
                            N,
                            N=N,
                            eps=float(eps),
                            root_size=float(self.root_size),
                            BLOCK_N=BLOCK_N,
                            num_warps=4 if BLOCK_N <= 2048 else 8,
                        )
                        out_proj = self.proj(out)
                        return out_proj[0] if isinstance(out_proj, tuple) else out_proj
                return orig_fn(self, x)
            return _fast_forward

        cls.forward = _make_forward(orig_forward)


register_vllm_patch("vllm.model_executor.models.gemma4", _patch_gemma4)
""",
    }
# EVOLVE-BLOCK-END
