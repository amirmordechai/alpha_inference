# Bring Your Own OpenAI `/v1/chat/completions` Trace

Drop any JSONL file of production requests (`messages` or `prompt`, plus optional `timestamp` in Unix seconds or ISO-8601) and point `optimize.py` at your model and GPU endpoint.

## JSONL Format (`sample_trace.jsonl`)
```json
{"request_id": "chat-0001", "timestamp": 1762160400.31, "messages": [{"role": "user", "content": "..."}], "max_tokens": 64}
```
- **With `timestamp`:** `alpha_inference` replays requests at exact **1:1 production arrival timing** so burst queueing is reproduced faithfully.
- **Without `timestamp`:** `alpha_inference` automatically falls back to controlled concurrent load (`c=24`).

## Run on Vertex AI Dedicated GPU Endpoints
```bash
python optimize.py \
  --trace examples/custom_openai_chat/sample_trace.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --deploy-vertex --workers 2 --auto-teardown \
  --metric ttft \
  --max-programs 40 \
  --budget-usd 30 \
  --out-dir runs/custom_chat
```

## Run on an Existing GPU VM
1. Start the hot-reload supervisor on your GPU VM:
   ```bash
   python -m alpha_inference.reload_proxy \
     --port 8080 --admin-token "$AE_ADMIN_TOKEN" \
     --model Qwen/Qwen2.5-7B-Instruct
   ```
2. Launch AlphaEvolve from your workstation:
   ```bash
   python optimize.py \
     --trace examples/custom_openai_chat/sample_trace.jsonl \
     --endpoint http://<GPU_VM_IP>:8080 --admin-token "$AE_ADMIN_TOKEN" \
     --model Qwen/Qwen2.5-7B-Instruct \
     --metric p99_latency \
     --max-programs 40 \
     --budget-usd 25 \
     --out-dir runs/custom_chat
   ```
