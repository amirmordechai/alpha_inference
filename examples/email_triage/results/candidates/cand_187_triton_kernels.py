"""Seed inference configuration evolved by Google Cloud AlphaEvolve (`candidate.py`).

AlphaEvolve mutates the function inside the evolve block below to discover faster
vLLM scheduler flags, CUDA graph capture ladders, runtime backends, and custom
Triton GPU kernels tailored to your workload trace and GPU hardware.
"""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
# IMPORTANT: Keep the function name `get_candidate_config` and return dict keys unchanged.
def get_candidate_config() -> dict[str, Any]:
    """Return the vLLM serving configuration and optional custom Triton GPU kernel."""
    return {
        "plain_title": "Tight Micro-Batching & V1 Cuda Graphs with Fast RMSNorm",
        "hypothesis": (
            "Switching back to V1 model runner (VLLM_USE_V2_MODEL_RUNNER=0) for better native CUDA "
            "graphs on 1-step diffusion workloads. Pushing micro-batching even tighter "
            "(--max-num-seqs=8, --max-num-batched-tokens=4096) to completely eliminate head-of-line "
            "blocking P99 spikes. Replacing softcap kernel with a fast Fused RMSNorm."
        ),
        "category": "triton_kernels",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
            "--served-model-name=diffusiongemma-26b",
            "--host=0.0.0.0",
            "--port=8080",
            '--diffusion-config={"canvas_length": 64}',
            "--max-logprobs=5",
            "--disable-uvicorn-access-log",
            "--attention-backend=TRITON_ATTN",
            "--max-num-seqs=8",
            "--max-num-batched-tokens=4096",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.92",
            "--async-scheduling",
            "--enable-prefix-caching",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
        },
        "custom_kernel_code": """
import torch
import triton
import triton.language as tl

@triton.jit
def fast_rmsnorm_kernel(
    Y_ptr, X_ptr, W_ptr,
    stride_y, stride_x,
    N, eps,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    Y_ptr += pid * stride_y
    X_ptr += pid * stride_x

    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)
    x_norm = x * rstd
    
    w = tl.load(W_ptr + offsets, mask=mask, other=1.0)
    out = x_norm * w
    tl.store(Y_ptr + offsets, out.to(w.dtype), mask=mask)

try:
    import vllm.model_executor.layers.layernorm as ln
    if not hasattr(ln.RMSNorm, '_original_forward_patched'):
        ln.RMSNorm._original_forward_patched = ln.RMSNorm.forward
        
        def fast_rmsnorm_forward(self, x):
            orig_shape = x.shape
            x_2d = x.view(-1, orig_shape[-1])
            M, N = x_2d.shape
            
            # Fallback for very large feature dims or non-contiguous tensors
            if not x_2d.is_contiguous() or N > 8192:
                return self._original_forward_patched(x)
                
            out = torch.empty_like(x_2d)
            BLOCK_N = triton.next_power_of_2(N)
            fast_rmsnorm_kernel[(M,)](
                out, x_2d, self.weight,
                out.stride(0), x_2d.stride(0),
                N, self.variance_epsilon,
                BLOCK_N=BLOCK_N,
                num_warps=4 if BLOCK_N <= 2048 else 8
            )
            return out.view(*orig_shape)
            
        ln.RMSNorm.forward = fast_rmsnorm_forward
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
