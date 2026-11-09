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
        "plain_title": "Native SM120 FlashInfer CUTLASS MoE + CuteDSL",
        "hypothesis": (
            "Instead of hand-maintaining a complex Triton MoE tile dictionary, delegating MoE and linear "
            "dispatch to native SM120 Blackwell kernels (flashinfer_cutlass / flashinfer_cutedsl with autotune) "
            "replaces generic Triton fused_moe_kernel (37.9% GPU time) and legacy sm80 wmma kernels (12.9%)."
        ),
        "category": "backends",  # vllm_flags | cuda_graphs | backends | triton_kernels
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
            '--kernel-config={"moe_backend":"flashinfer_cutlass","linear_backend":"flashinfer_cutedsl","enable_flashinfer_autotune":true}',
            '--compilation-config={"max_cudagraph_capture_size":512}',
            "--disable-uvicorn-access-log",
        ],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
