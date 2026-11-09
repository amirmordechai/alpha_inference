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
        "plain_title": "Phase-Split MoE Tiles + Fused Router Triton",
        "hypothesis": (
            "Bifurcating moe_tuned_config across micro-decode (M16N128, GROUP_SIZE_M=1), "
            "high-decode (M16N128, GROUP_SIZE_M=4), and prefill (M32/64N128, 8 warps), "
            "plus a dense CUDA graph ladder and fused Triton Gemma4 router norm slashes P99."
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
            "--max-num-batched-tokens=4096",
            "--long-prefill-token-threshold=1024",
            "--quantization=fp8",
            "--language-model-only",
            "--async-scheduling",
            '--kernel-config={"enable_flashinfer_autotune":true}',
            (
                '--compilation-config={"max_cudagraph_capture_size":512,'
                '"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,320,384,448,512]}'
            ),
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            **{
                m: {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 4,
                }
                for m in (1, 2, 4, 8, 16, 32, 64)
            },
            **{
                m: {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 4,
                    "num_warps": 4,
                    "num_stages": 4,
                }
                for m in (128, 256)
            },
            **{
                m: {
                    "BLOCK_SIZE_M": 32 if m <= 1024 else 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                    "num_warps": 8,
                    "num_stages": 4,
                }
                for m in (512, 1024, 2048, 4096, 8192)
            },
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl
from sitecustomize import register_vllm_patch


@triton.jit
def _gemma4_router_norm_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    stride_x_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    root_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps) * root_size
    y = (x * rstd) * s
    tl.store(out_ptr + row * stride_x_row + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


def _patch_gemma4_model(mod):
    for attr_name in dir(mod):
        cls = getattr(mod, attr_name, None)
        if not isinstance(cls, type):
            continue
        if "Router" in attr_name and hasattr(cls, "forward"):
            orig_fwd = cls.forward

            def _make_fast_fwd(orig):
                def _fast_forward(self, x):
                    if (
                        hasattr(self, "scale")
                        and hasattr(self, "root_size")
                        and hasattr(self, "eps")
                        and hasattr(self, "proj")
                        and x.is_cuda
                        and x.is_contiguous()
                    ):
                        orig_shape = x.shape
                        N = orig_shape[-1]
                        BLOCK_N = triton.next_power_of_2(N)
                        if BLOCK_N <= 8192:
                            x_2d = x.view(-1, N)
                            M = x_2d.shape[0]
                            out = torch.empty_like(x_2d)
                            num_warps = 4 if BLOCK_N <= 2048 else 8
                            _gemma4_router_norm_scale_kernel[(M,)](
                                x_2d,
                                self.scale,
                                out,
                                x_2d.stride(0),
                                N=N,
                                eps=float(self.eps),
                                root_size=float(self.root_size),
                                BLOCK_N=BLOCK_N,
                                num_warps=num_warps,
                            )
                            logits = self.proj(out.view(orig_shape))
                            if hasattr(self, "topk"):
                                return self.topk(logits)
                    return orig(self, x)

                return _fast_forward

            cls.forward = _make_fast_fwd(orig_fwd)


for _mod_name in (
    "vllm.model_executor.models.gemma4",
    "vllm.model_executor.models.gemma3",
    "vllm.model_executor.models.gemma3_moe",
):
    register_vllm_patch(_mod_name, _patch_gemma4_model)
""",
    }
# EVOLVE-BLOCK-END
