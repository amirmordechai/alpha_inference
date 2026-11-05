# GenAI Inference & GPU Kernel Optimization on vLLM

Evolve `get_candidate_config()` inside `candidate.py` to optimize live `vLLM` inference performance on a real 1:1 production request trace while passing the strict Layer-2 Token-Logprob KL Output-Equivalence Gate.

## Hard Contract (Never Violate)
1. **Do NOT rename `get_candidate_config()`** or remove any returned dictionary keys:
   `plain_title` (short <=48-char human summary, no raw CLI flags), `hypothesis`, `category` (`vllm_flags` | `cuda_graphs` | `backends` | `triton_kernels`), `vllm_args` (`list[str]`), `env_vars` (`dict[str, str]`), `custom_kernel_code` (`str`).
2. **Immutable Model & Precision:** Keep the exact model ID (`vllm_args[0]`), `--served-model-name`, `--max-model-len >= baseline`, and `--max-logprobs >= 5`. Never swap checkpoints, drop `--diffusion-config` (if present in baseline), or enable lossy KV quantization (`--kv-cache-dtype=fp8`).
3. **Deterministic Container Compatibility Guards (`vLLM v0.31+` / CUDA 13):**
   - Do not pass `--disable-log-requests` (removed in `vLLM v0.31+`; use `--disable-uvicorn-access-log`).
   - Do not set `CUDA_MODULE_LOADING="EAGER"` (triggers multi-minute PTX JIT compilation stalls on Blackwell `sm_120`) or `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` (conflicts with `vLLM` CUDA graph memory pools).
4. **Scientific Incremental Evolution:** Mutate **1 to 2 orthogonal levers at a time** from the fastest working parent program so beneficial changes compound cleanly.

## Optimization Search Space
1. **`vllm_flags` (Continuous Batching, Chunked Prefill, Prefix Caching & Scheduler):**
   - Tune `--max-num-seqs` (e.g. `64`..`256` — avoid starving concurrency under high arrival rates!), `--max-num-batched-tokens` (`4096`..`16384`), and `--long-prefill-token-threshold` (`512`..`4096`) to balance prefill-decode interference, queueing delay, and step latency for the active workload utilization.
   - Explore `--enable-prefix-caching`, `--async-scheduling`, `--max-logprobs` (down to `5`), `--disable-uvicorn-access-log`, `--gpu-memory-utilization` (`0.90`..`0.95`), KV `--block-size`, and (for multimodal checkpoints serving text-only traces) `--language-model-only` (note: pass `--language-model-only` before lowering `--max-num-batched-tokens` below `8192`, otherwise multimodal encoder token budgets raise `ValueError` at startup).
2. **`cuda_graphs` (`--compilation-config`):**
   - Tune `cudagraph_capture_sizes` ladders across active decode/prefill batch sizes, `max_cudagraph_capture_size` (`512`, `1024`, `2048`, `4096`), and `pass_config` / `inductor_compile_config` fusions (avoid deprecated `"level"` keys in `vLLM v0.31+`).
3. **`backends` (Attention, MoE, Tokenizer & Host-Thread Runtime):**
   - Explore `--attention-backend` (`FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`; note `FLASHINFER`/`FLASH_ATTN` only support `head_dim <= 256`, so checkpoints with `head_dim=512` full-attention layers or `--diffusion-config` require `TRITON_ATTN`), `TOKENIZERS_PARALLELISM="true"`, `OMP_NUM_THREADS` (`2`..`8`), `CUDA_DEVICE_MAX_CONNECTIONS`, and `VLLM_USE_V2_MODEL_RUNNER`.
4. **`triton_kernels` (`custom_kernel_code` via `sitecustomize.py`):**
   - `custom_kernel_code` executes via `sitecustomize.py` before `vLLM` worker spawn so `@triton.jit` kernels are captured directly inside CUDA graphs. Always wrap monkey-patches in defensive `try: ... except Exception: pass` guards and never import network or subprocess modules.
   - Target hot GPU ops surfaced in the live Kineto profile (e.g., fused `RMSNorm` + residual add, `tanh` logit softcapping in `tl.float32` cast back to `bf16`, RoPE, SwiGLU/GeGLU epilogues, or MoE routing/reduction helpers).
