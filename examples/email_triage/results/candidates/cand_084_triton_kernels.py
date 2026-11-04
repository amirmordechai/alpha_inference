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
        "plain_title": "Extreme Micro-Batching & Crazy Triton RMSNorm",
        "hypothesis": (
            "Over-batching (max-num-seqs=24) caused massive head-of-line blocking. "
            "We go crazy with extreme micro-batching (max-num-seqs=6, max-num-batched-tokens=3072) "
            "to ensure instantaneous step drain, increase OMP threads to 8, and inject "
            "a wildly aggressive Triton RMSNorm kernel monkey-patch."
        ),
        "category": "triton_kernels",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
            "--served-model-name=diffusiongemma-26b",
            "--host=0.0.0.0",
            "--port=8080",
            "--disable-uvicorn-access-log",
            '--diffusion-config={"canvas_length": 64}',
            "--max-logprobs=5",
            '--override-generation-config={"max_new_tokens": null}',
            "--enable-auto-tool-choice",
            "--tool-call-parser=gemma4",
            "--reasoning-parser=gemma4",
            "--enable-prefix-caching",
            "--async-scheduling",
            '--limit-mm-per-prompt={"image": 0, "video": 0}',
            "--attention-backend=TRITON_ATTN",
            "--max-num-seqs=6",
            "--max-num-batched-tokens=3072",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
        },
        "custom_kernel_code": """
try:
    import torch
    import triton
    import triton.language as tl
    from vllm.model_executor.layers.layernorm import RMSNorm

    @triton.jit
    def crazy_fast_rmsnorm(Y, X, W, stride_y, stride_x, N, eps, BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        Y += row * stride_y
        X += row * stride_x
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rsqrt = tl.math.rsqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rsqrt * w
        tl.store(Y + cols, y.to(tl.bfloat16), mask=mask)

    def crazy_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        x_2d = x.view(-1, x.shape[-1])
        out_2d = out.view(-1, out.shape[-1])
        N = x_2d.shape[1]
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 2048 else 8
        crazy_fast_rmsnorm[(x_2d.shape[0],)](
            out_2d, x_2d, self.weight, out_2d.stride(0), x_2d.stride(0),
            N, self.variance_epsilon, BLOCK_N=BLOCK_N, num_warps=num_warps
        )
        return out

    RMSNorm.forward = crazy_forward
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
