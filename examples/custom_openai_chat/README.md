# Bring Your Own OpenAI `/v1/chat/completions` Trace

Drop any JSONL file of production requests (`messages` or `prompt`, plus optional `timestamp` in Unix seconds or ISO-8601) and point `optimize.py` at your live Vertex AI GPU Endpoint or GPU VM.

## JSONL Format (`sample_trace.jsonl`)
```json
{"request_id": "chat-0001", "timestamp": 1762160400.6375, "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "max_tokens": 64, "temperature": 0.0}
```
- **With `timestamp`:** `alpha_inference` replays requests at exact **1:1 production arrival timing** so burst queueing is reproduced faithfully.
- **Without `timestamp`:** `alpha_inference` automatically falls back to controlled concurrent load (`c=24`).

## 1. Optimize Any Live Vertex AI GPU Endpoint
`optimize.py` inspects `$VERTEX_ENDPOINT_ID`, automatically inherits its GPU type, machine shape, container image, and model, spins up temporary warm hot-reload search workers, and tears them down on exit:

```bash
python optimize.py \
  --vertex-endpoint-id "$VERTEX_ENDPOINT_ID" \
  --trace examples/custom_openai_chat/sample_trace.jsonl \
  --metric p99_latency \
  --max-programs 60 \
  --budget-usd 40 \
  --out-dir results/custom_chat
```

## 2. Or Optimize on an Existing GPU VM
1. Start the hot-reload supervisor on your GPU VM:
   ```bash
   python -m alpha_inference.reload_proxy \
     --port 8080 --admin-token "$AE_ADMIN_TOKEN" \
     --model google/gemma-4-26B-A4B-it
   ```
2. Launch AlphaEvolve from your workstation:
   ```bash
   python optimize.py \
     --trace examples/custom_openai_chat/sample_trace.jsonl \
     --endpoint "http://<GPU_VM_IP>:8080" --admin-token "$AE_ADMIN_TOKEN" \
     --model google/gemma-4-26B-A4B-it \
     --metric p99_latency \
     --max-programs 40 \
     --budget-usd 25 \
     --out-dir results/custom_chat
   ```
