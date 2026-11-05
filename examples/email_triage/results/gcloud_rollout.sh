#!/usr/bin/env bash
set -euo pipefail

# Roll out the AlphaEvolve winning configuration using native Vertex AI (`gcloud ai`) CLI:
MODEL_ID=$(gcloud ai models upload \
  --project=${GCP_PROJECT_ID} \
  --region=us-central1 \
  --display-name=alpha-inference-best \
  --container-image-uri=docker.io/vllm/vllm-openai:v0.31.0 \
  --container-command='bash,-lc,mkdir -p /tmp/ae_kernel && python3 -c "import base64, os, pathlib; pathlib.Path('"'"'/tmp/ae_kernel/sitecustomize.py'"'"').write_bytes(base64.b64decode(os.environ['"'"'AE_KERNEL_B64'"'"']))" && export PYTHONPATH="/tmp/ae_kernel:${PYTHONPATH:-}" && exec vllm serve "$@",--' \
  --container-args='^||^google/gemma-4-26B-A4B-it||--served-model-name=google/gemma-4-26B-A4B-it||--host=0.0.0.0||--port=8080||--max-model-len=8192||--max-logprobs=5||--gpu-memory-utilization=0.95||--attention-backend=flashinfer||--enable-chunked-prefill||--language-model-only||--max-num-batched-tokens=2048||--max-num-seqs=256||--block-size=32||--enable-prefix-caching||--disable-uvicorn-access-log||--async-scheduling' \
  --container-env-vars='^||^PYTHONUNBUFFERED=1||TOKENIZERS_PARALLELISM=true||OMP_NUM_THREADS=8||CUDA_DEVICE_MAX_CONNECTIONS=1||VLLM_USE_V2_MODEL_RUNNER=1||VLLM_LOGGING_LEVEL=WARNING||HF_HOME=/dev/shm/hf_cache||AE_KERNEL_B64=CnRyeToKICAgIGltcG9ydCBnYwogICAgZ2MuZGlzYWJsZSgpCmV4Y2VwdCBFeGNlcHRpb246CiAgICBwYXNzCgp0cnk6CiAgICBpbXBvcnQgb3MKICAgIG9zLm5pY2UoLTEwKQpleGNlcHQgRXhjZXB0aW9uOgogICAgcGFzcwoKdHJ5OgogICAgaW1wb3J0IHRvcmNoCiAgICB0b3JjaC5zZXRfZ3JhZF9lbmFibGVkKEZhbHNlKQogICAgdG9yY2guc2V0X251bV90aHJlYWRzKDgpCiAgICB0b3JjaC5iYWNrZW5kcy5jdWRhLm1hdG11bC5hbGxvd190ZjMyID0gVHJ1ZQogICAgdG9yY2guYmFja2VuZHMuY3Vkbm4uYWxsb3dfdGYzMiA9IFRydWUKZXhjZXB0IEV4Y2VwdGlvbjoKICAgIHBhc3MK' \
  --container-ports=8080 \
  --container-predict-route=/v1/chat/completions \
  --container-health-route=/health \
  --container-shared-memory-size-mb=83968 \
  --container-deployment-timeout-seconds=3600 \
  --format='value(model)' --quiet)

gcloud ai endpoints deploy-model ${VERTEX_ENDPOINT_ID} \
  --project=${GCP_PROJECT_ID} \
  --region=us-central1 \
  --model="$MODEL_ID" \
  --display-name=alpha-inference-best \
  --machine-type=g4-standard-48 \
  --accelerator=type=nvidia-rtx-pro-6000,count=1 \
  --min-replica-count=1 --max-replica-count=1 \
  --traffic-split=0=100 --quiet
