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
        "plain_title": "16x128x128 FP8 MoE tiles + dense spec cudagraphs",
        "hypothesis": (
            "With 3-token ngram_gpu speculation, decode batches span M=256..512 where "
            "E=128, top_k=8 yields only 16..32 tokens/expert. Using 16x128x128 MoE tiles "
            "up to M=512 and 32-step CUDA graphs in 256..512 cuts MoE and padding overhead."
        ),
        "category": "cuda_graphs",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": [
            "google/gemma-4-26B-A4B-it",
            "--served-model-name=google/gemma-4-26B-A4B-it",
            "--host=0.0.0.0",
            "--port=8080",
            "--max-model-len=8192",
            "--max-logprobs=16",
            "--gpu-memory-utilization=0.94",
            "--quantization=fp8",
            "--async-scheduling",
            "--language-model-only",
            "--max-num-seqs=256",
            "--max-num-batched-tokens=8192",
            "--long-prefill-token-threshold=1024",
            '--speculative-config={"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}',
            '--kernel-config={"enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":512,"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512]}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "moe_tuned_config": {
            m: {
                "BLOCK_SIZE_M": 16 if m <= 512 else 32,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1 if m <= 512 else 8,
                "num_warps": 8,
                "num_stages": 4,
            }
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
