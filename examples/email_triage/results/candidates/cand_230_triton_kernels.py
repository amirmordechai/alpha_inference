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
        "plain_title": "3-Gram Spec + M512 Zero-Pad MoE + Fused Router",
        "hypothesis": (
            "Restoring ngram_gpu with a stricter 3-to-6 token lookup window raises speculative acceptance "
            "precision, while extending BLOCK_SIZE_M=16/N=64 tiles and CUDA graphs to M=512 plus fusing "
            "Gemma-4 router RMSNorm+scale absorbs the 4x speculative token batch expansion."
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
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_max":6,"prompt_lookup_min":3}',
            '--kernel-config={"enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":512,"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,448,512]}',
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            m: {
                "BLOCK_SIZE_M": 16 if m <= 512 else 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1 if m <= 32 else 8,
                "num_warps": 4,
                "num_stages": 4,
            }
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
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
    stride_om,
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
    tl.store(out_ptr + row * stride_om + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


def _fused_router_norm_scale(x: torch.Tensor, scale: torch.Tensor, eps: float, root_size: float) -> torch.Tensor:
    orig_shape = x.shape
    N = orig_shape[-1]
    x_2d = x.contiguous().view(-1, N)
    M = x_2d.shape[0]
    out = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    _fused_router_norm_scale_kernel[(M,)](
        x_2d,
        scale,
        out,
        x_2d.stride(0),
        out.stride(0),
        N=N,
        eps=eps,
        root_size=root_size,
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N <= 4096 else 8,
    )
    return out.view(orig_shape)


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
                    ):
                        eps = getattr(self.norm, "variance_epsilon", 1e-6)
                        h = _fused_router_norm_scale(x, self.scale, float(eps), float(self._root_size))
                        router_logits, _ = self.proj(h)
                        return self.topk(router_logits)
                    return orig_fn(self, x)

                return _router_forward

            cls.forward = _make_router_forward(orig_forward)


register_vllm_patch("vllm.model_executor.models.gemma4", _patch_gemma4)
""",
    }
# EVOLVE-BLOCK-END
