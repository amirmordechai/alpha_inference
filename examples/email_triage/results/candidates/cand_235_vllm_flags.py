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
    import inspect
    
    # Base arguments that are always safe and present in the best config
    vllm_args = [
        "google/gemma-4-26B-A4B-it",
        "--served-model-name=google/gemma-4-26B-A4B-it",
        "--host=0.0.0.0",
        "--port=8080",
        "--max-model-len=8192",
        "--max-logprobs=5",
        "--gpu-memory-utilization=0.95",
        "--attention-backend=flashinfer",
        "--language-model-only",
        "--max-num-seqs=256",
        "--block-size=32",
        "--enable-prefix-caching",
        "--disable-uvicorn-access-log",
        "--async-scheduling",
    ]
    
    # Try to dynamically inspect EngineArgs to use advanced features safely
    try:
        from vllm.engine.arg_utils import EngineArgs
        params = inspect.signature(EngineArgs).parameters
    except Exception:
        params = {}
        
    # Dynamic Chunked Prefill tuning
    # We use 4096 batched tokens and 1024 long prefill threshold to prevent decode starvation
    if "enable_chunked_prefill" in params:
        vllm_args.append("--enable-chunked-prefill")
        vllm_args.append("--max-num-batched-tokens=4096")
        if "long_prefill_token_threshold" in params:
            vllm_args.append("--long-prefill-token-threshold=1024")
    elif "chunked_prefill" in params:
        vllm_args.append("--chunked-prefill")
        vllm_args.append("--max-num-batched-tokens=4096")
        if "long_prefill_token_threshold" in params:
            vllm_args.append("--long-prefill-token-threshold=1024")
    else:
        # Fallback to standard chunked prefill arguments if introspection fails
        vllm_args.append("--enable-chunked-prefill")
        vllm_args.append("--max-num-batched-tokens=4096")
        vllm_args.append("--long-prefill-token-threshold=1024")
        
    # Multi-step scheduling can drastically reduce CPU overhead and step latency
    if "num_scheduler_steps" in params:
        vllm_args.append("--num-scheduler-steps=4")
        
    # V2 block manager can improve memory allocation performance
    if "use_v2_block_manager" in params:
        vllm_args.append("--use-v2-block-manager")

    # Sampler backend optimization
    if "sampler_backend" in params:
        vllm_args.append("--sampler-backend=flashinfer")
        
    return {
        "plain_title": "Tuned Chunked Prefill & Sweet-Spot Multi-Step",
        "hypothesis": (
            "Reducing num-scheduler-steps to 4 to minimize scheduling delay/TTFT of incoming requests, "
            "and lowering long-prefill-token-threshold to 1024 to chunk long prefills earlier, "
            "preventing decode starvation and optimizing P99 latency."
        ),
        "category": "vllm_flags",  # vllm_flags | cuda_graphs | backends | triton_kernels
        "vllm_args": vllm_args,
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
        },
        "custom_kernel_code": "",
    }
# EVOLVE-BLOCK-END
