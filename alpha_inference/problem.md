# GenAI Serving & GPU Kernel Optimization on vLLM

Evolve `get_candidate_config()` inside `candidate.py` to optimize live `vLLM` (v0.31.0 V1/V2 engine) serving performance on a real 1:1 production request trace while passing the strict Layer-2 Token-Logprob KL Output-Equivalence Gate.

## Hard Contract (Never Violate)
1. **Do NOT rename `get_candidate_config()`** or remove any returned dictionary keys:
   `plain_title` (short <=48-char human summary, no raw CLI flags), `hypothesis`, `category` (`vllm_flags` | `cuda_graphs` | `backends` | `triton_kernels`), `vllm_args` (`list[str]`), `env_vars` (`dict[str, str]`), `custom_kernel_code` (`str`).
2. **Immutable Model & Precision:** Keep the exact model ID (`vllm_args[0]`), `--served-model-name`, `--max-model-len >= baseline`, and `--max-logprobs >= 5`. Never swap checkpoints, drop `--diffusion-config` (if present), or enable lossy KV quantization (`--kv-cache-dtype=fp8`).
3. **vLLM v0.31 & Single-GPU Blackwell (`sm_120`, `TP=1`) Compatibility Rules:**
   - **NEVER** pass `--disable-log-requests` (removed in vLLM v0.31; use `--disable-uvicorn-access-log`).
   - **NEVER** set `CUDA_MODULE_LOADING="EAGER"` (triggers 8+ min PTX JIT compilation timeouts on Blackwell `sm_120`), `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`, or `CUDA_DEVICE_MAX_CONNECTIONS="1"` (serializes async H2D copies and compute streams on `TP=1`, cutting throughput by ~20%).
4. **Scientific Incremental Evolution:** Mutate **1 to 2 orthogonal levers at a time** from the fastest working parent program rather than changing 6 parameters simultaneously. That way beneficial changes compound cleanly instead of being masked by regressions.

## High-Leverage Optimization Space
1. **`vllm_flags` (Micro-Batching vs. Head-of-Line Blocking, Prefill Budget & Output Overhead):**
   - **Critical P99 Physics:** Under bursty 1:1 production arrivals, **over-batching** (`--max-num-seqs >= 64` or `--max-num-batched-tokens >= 16384`) packs dozens of sequences into one slow mega-batch, causing head-of-line blocking that *hurts* P99 latency! Explore **tighter and moderate** micro-batch limits (`--max-num-seqs` across `6`, `8`, `10`, `12`, `14`, `16`, `20`, `24`, `32`) and `--max-num-batched-tokens` (`2048`, `3072`, `4096`, `6144`, `8192`, `12288`) to find the exact sweet spot between fast step drain and burst absorption.
   - Note on 1-step diffusion (`--diffusion-config={"canvas_length": 64}`): each request completes in a single forward pass (`prompt_tokens + 64`), so small `--long-prefill-token-threshold` values split long prompts into extra forward passes and add overhead — prefer omitting `--long-prefill-token-threshold` or keeping it large (`>= 4096`).
   - Reduce output/logging overhead: lower `--max-logprobs` from `32` down to `5`, add `--disable-uvicorn-access-log`, and tune `--gpu-memory-utilization` (`0.90`..`0.95`), `--enable-prefix-caching`, `--async-scheduling`.
2. **`cuda_graphs` (`--compilation-config`):**
   - In vLLM v0.31 V1, `FULL` CUDA graphs are captured for small token budgets (up to `512` by default) and `PIECEWISE` graphs above that, and request token counts (`prompt_tokens + 64`) are arbitrary integers — **do NOT drop small capture sizes (`1, 2, 4, 8, 16, 24, 32, ...`) in favor of coarse multiples of 64**!
   - Explore keeping default capture sizes vs. denser small-to-medium token ladders (e.g. `[1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]`), increasing `max_cudagraph_capture_size` (`512`, `1024`, `2048`), or tuning `pass_config` / `inductor_compile_config` fusions (never pass deprecated `"level"` keys).
3. **`backends` (Tokenizer & Host-Thread Env Vars):**
   - Tune `TOKENIZERS_PARALLELISM="true"`, `OMP_NUM_THREADS` (`4` or `8`), `VLLM_USE_V2_MODEL_RUNNER="1"`. Keep default `CUDA_DEVICE_MAX_CONNECTIONS` (`8`, i.e. omit it) so H2D copy and compute streams overlap on `TP=1`.
   - If `--diffusion-config` is present (`DiffusionGemma`), bidirectional masks require `--attention-backend=TRITON_ATTN`.
4. **`triton_kernels` (`custom_kernel_code` via `sitecustomize.py`):**
   - `custom_kernel_code` runs via `sitecustomize.py` at startup so GPU workers compile and capture your `@triton.jit` ops inside CUDA graphs. Always wrap monkey-patches in defensive `try: ... except Exception: pass` guards and never import `socket`, `urllib`, `requests`, `aiohttp`, or `subprocess`.
   - Build custom `@triton.jit` fusions **directly on top of fast low-latency scheduler configs** (not slow `max-num-seqs=128` configs!), targeting memory-bound elementwise ops such as `tanh` logit softcapping over `262,144` vocab logits (`x = tanh(x / cap) * cap` computed in `tl.float32` then cast back to input `bf16`), `RMSNorm` / fused residual `RMSNorm`, or RoPE / top-k logprob reductions.
