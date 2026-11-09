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
        "plain_title": "Drop Low-Accept Spec + Dual Fused Gemma4 Triton",
        "hypothesis": (
            "At 8.7% acceptance (1.17 tok/step), ngram_gpu triples MoE token batch M for only 17% extra tokens; "
            "removing it shrinks MoE decode batches by 3x while fusing both Router RMSNorm+scale and LayerScalar+Residual in Triton."
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
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=2048",
            '--kernel-config={"enable_flashinfer_autotune":true}',
            '--compilation-config={"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256],"max_cudagraph_capture_size":256}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            1: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
            2: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
            4: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
            8: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
            16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
            32: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
            64: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
            128: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
            256: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3},
            512: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            1024: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl
from sitecustomize import register_vllm_patch

@triton.jit
def _fused_router_norm_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    stride_x,
    N,
    eps,
    root_size,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = tl.rsqrt(var + eps)
    y = x * rrms * root_size * s
    tl.store(out_ptr + row * stride_x + cols, y.to(x_ptr.dtype.element_ty), mask=mask)

def _fused_router_input_scale(x: torch.Tensor, scale: torch.Tensor, eps: float, root_size: float) -> torch.Tensor:
    orig_shape = x.shape
    N = orig_shape[-1]
    x_2d = x.contiguous().view(-1, N)
    M = x_2d.shape[0]
    out = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_N <= 1024 else 8
    _fused_router_norm_scale_kernel[(M,)](
        x_2d, scale, out,
        x_2d.stride(0),
        N,
        eps,
        root_size,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out.view(orig_shape)

@triton.jit
def _fused_rmsnorm_weight_scalar_kernel(
    x_ptr, w_ptr, scalar_ptr, out_ptr,
    stride_x,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    sc = tl.load(scalar_ptr).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = tl.rsqrt(var + eps)
    y = x * rrms * w * sc
    tl.store(out_ptr + row * stride_x + cols, y.to(x_ptr.dtype.element_ty), mask=mask)

def _ patch_rmsnorm_with_scalar(norm_mod, scalar_tensor: torch.Tensor):
    orig_norm_fwd = norm_mod.forward
    eps = float(getattr(norm_mod, "variance_epsilon", 1e-6))
    def _fused_norm_fwd(x: torch.Tensor, residual=None):
        if residual is not None or getattr(norm_mod, "weight", None) is None:
            return orig_norm_fwd(x, residual) if residual is not None else orig_norm_fwd(x)
        orig_shape = x.shape
        N = orig_shape[-1]
        x_2d = x.contiguous().view(-1, N)
        M = x_2d.shape[0]
        out = torch.empty_like(x_2d)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _fused_rmsnorm_weight_scalar_kernel[(M,)](
            x_2d, norm_mod.weight, scalar_tensor, out,
            x_2d.stride(0),
            N,
            eps,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
    norm_mod.forward = _fused_norm_fwd

def _patch_gemma4(mod):
    for attr_name in dir(mod):
        cls = getattr(mod, attr_name, None)
        if not isinstance(cls, type):
            continue
        if "Router" in attr_name and hasattr(cls, "forward"):
            orig_fwd = cls.forward
            def _make_router_fwd(orig):
                def _fwd(self, x):
                    if (
                        hasattr(self, "norm")
                        and hasattr(self, "scale")
                        and hasattr(self, "root_size")
                        and hasattr(self, "proj")
                        and getattr(self.norm, "weight", None) is None
                    ):
                        eps = getattr(self.norm, "variance_epsilon", 1e-6)
                        h = _fused_router_input_scale(x, self.scale, float(eps), float(self.root_size))
                        router_logits, _ = self.proj(h)
                        return router_logits
                    return orig(self, x)
                return _fwd
            cls.forward = _make_router_fwd(orig_fwd)
        if "DecoderLayer" in attr_name and hasattr(cls, "__init__"):
            orig_init = cls.__init__
            def _make_layer_init(orig):
                def _init(self, *args, **kwargs):
                    orig(self, *args, **kwargs)
                    if hasattr(self, "layer_scalar") and hasattr(self, "post_feedforward_layernorm"):
                        _patch_rmsnorm_with_scalar(self.post_feedforward_layernorm, self.layer_scalar)
                        self._ones_scalar = torch.ones_like(self.layer_scalar)
                        self.layer_scalar = torch.nn.Parameter(self._ones_scalar, requires_grad=False)
                return _init
            cls.__init__ = _make_layer_init(orig_init)

register_vllm_patch("vllm.model_executor.models.gemma4", _patch_gemma4)
""",
    }
# EVOLVE-BLOCK-END
