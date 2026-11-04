#!/usr/bin/env bash
set -euo pipefail

# Roll out the AlphaEvolve winning configuration using native Vertex AI (`gcloud ai`) CLI:
MODEL_ID=$(gcloud ai models upload \
  --project=${GCP_PROJECT_ID} \
  --region=${GCP_REGION:-us-central1} \
  --display-name=alpha-inference-best \
  --container-image-uri=${VLLM_IMAGE_URI} \
  --container-command=vllm,serve \
  --container-args='^||^nvidia/diffusiongemma-26B-A4B-it-NVFP4||--served-model-name=diffusiongemma-26b||--host=0.0.0.0||--port=8080||--disable-uvicorn-access-log||--diffusion-config={"canvas_length": 64}||--max-logprobs=5||--override-generation-config={"max_new_tokens": null}||--enable-auto-tool-choice||--tool-call-parser=gemma4||--reasoning-parser=gemma4||--enable-prefix-caching||--async-scheduling||--limit-mm-per-prompt={"image": 0, "video": 0}||--attention-backend=TRITON_ATTN||--max-num-seqs=24||--max-num-batched-tokens=4096||--max-model-len=12288||--gpu-memory-utilization=0.90||--compilation-config={"cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048], "max_cudagraph_capture_size": 2048}' \
  --container-env-vars='^||^VLLM_USE_V2_MODEL_RUNNER=1||PYTHONUNBUFFERED=1||TOKENIZERS_PARALLELISM=true||OMP_NUM_THREADS=4' \
  --container-ports=8080 \
  --container-predict-route=/v1/chat/completions \
  --container-health-route=/health \
  --format='value(model)' --quiet)

gcloud ai endpoints deploy-model ${VERTEX_ENDPOINT_ID} \
  --project=${GCP_PROJECT_ID} \
  --region=${GCP_REGION:-us-central1} \
  --model="$MODEL_ID" \
  --display-name=alpha-inference-best \
  --machine-type=g4-standard-48 \
  --accelerator=accelerator-type=nvidia-rtx-pro-6000,count=1 \
  --min-replica-count=1 --max-replica-count=1 \
  --traffic-split=0=100 --quiet
