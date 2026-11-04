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
        "plain_title": "Fused Softcapping & Dense CUDAGraphs",
        "hypothesis": (
            "Intercept LogitsProcessor with a custom in-place Triton kernel to fuse memory-bound "
            "tanh softcapping, saving VRAM bandwidth on the 256k vocab. Pair this with a dense "
            "CUDAGraph capture ladder up to 4096 tokens to perfectly cover dynamic batch sizes."
        ),
        "category": "triton_kernels",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
            "--served-model-name=diffusiongemma-26b",
            "--host=0.0.0.0",
            "--port=8080",
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
            "--max-num-seqs=10",
            "--max-num-batched-tokens=4096",
            "--max-model-len=12288",
            "--gpu-memory-utilization=0.95",
            "--disable-uvicorn-access-log",
            '--compilation-config={"max_cudagraph_capture_size": 4096, "cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]}',
        ],
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "true",
        },
        "custom_kernel_code": """
try:
    import torch
    import triton
    import triton.language as tl
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    @triton.jit
    def softcap_kernel_2d(
        logits_ptr,
        stride_tok, stride_vocab,
        n_tokens, n_vocab,
        cap,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        n_elements = n_tokens * n_vocab
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        tok_idx = offsets // n_vocab
        vocab_idx = offsets % n_vocab
        
        load_ptrs = logits_ptr + tok_idx * stride_tok + vocab_idx * stride_vocab
        
        x = tl.load(load_ptrs, mask=mask)
        x_f32 = x.to(tl.float32)
        x_f32 = x_f32 / cap
        x_f32 = tl.math.tanh(x_f32)
        x_f32 = x_f32 * cap
        
        tl.store(load_ptrs, x_f32.to(x.dtype), mask=mask)

    orig_forward = LogitsProcessor.forward

    def fast_forward(self, *args, **kwargs):
        cap = getattr(self, "soft_cap", None)
        if cap is not None:
            self.soft_cap = None
            try:
                logits = orig_forward(self, *args, **kwargs)
            finally:
                self.soft_cap = cap
            
            if logits.dim() == 2:
                n_tokens, n_vocab = logits.shape
                stride_tok, stride_vocab = logits.stride()
                grid = lambda meta: (triton.cdiv(n_tokens * n_vocab, meta['BLOCK_SIZE']),)
                softcap_kernel_2d[grid](
                    logits, stride_tok, stride_vocab, n_tokens, n_vocab, cap, BLOCK_SIZE=1024
                )
            else:
                logits = logits / cap
                logits = torch.tanh(logits)
                logits = logits * cap
            return logits
        else:
            return orig_forward(self, *args, **kwargs)

    LogitsProcessor.forward = fast_forward
except Exception:
    pass
""",
    }
# EVOLVE-BLOCK-END
