#!/usr/bin/env bash
set -euo pipefail

# DiffusionGemma 26B A4B IT NVFP4 on DGX Spark / GB10
# Official gemma image + NVIDIA ModelOpt NVFP4 checkpoint

MODEL_ID=${MODEL_ID:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}
IMAGE=${IMAGE:-vllm/vllm-openai:gemma-aarch64-cu130}
PORT=${PORT:-8000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-10}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
NAME=${NAME:-diffusiongemma-26b-nvfp4}
MODEL_DIR=${MODEL_DIR:-$HOME/llm/diffusiongemma-26b-a4b-it-nvfp4}

export PATH="$HOME/.local/bin:$PATH"

if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "Downloading $MODEL_ID -> $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  if [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
  fi
  hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
fi

docker pull "$IMAGE" || true
docker rm -f "$NAME" 2>/dev/null || true

# Entrypoint is already [vllm, serve]
docker run -d --name "$NAME" --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -v "$MODEL_DIR:/models/diffusiongemma:ro" \
  "$IMAGE" \
  /models/diffusiongemma \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --attention-backend TRITON_ATTN \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --override-generation-config "{\"max_new_tokens\": null}" \
    --default-chat-template-kwargs "{\"enable_thinking\": true}" \
    --served-model-name "$MODEL_ID" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN"

echo "Waiting for API on :$PORT ..."
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "Ready: http://127.0.0.1:$PORT/v1"
    curl -s "http://127.0.0.1:$PORT/v1/models"
    exit 0
  fi
  if [[ "$(docker inspect -f "{{.State.Status}}" "$NAME" 2>/dev/null || echo missing)" != "running" ]]; then
    echo "Container died:"
    docker logs "$NAME" 2>&1 | tail -80
    exit 1
  fi
  sleep 5
done
echo "Timeout waiting for API"
docker logs "$NAME" 2>&1 | tail -80
exit 1
