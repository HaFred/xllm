#!/bin/bash
set -euo pipefail

IMAGE="${XLLM_IMAGE:-quay.io/jd_xllm/xllm-ai:xllm-dev-cuda-x86}"
XLLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${XLLM_CONTAINER_NAME:-xllm-cuda}"
MODEL_HOST="${XLLM_MODEL_HOST:-/scratch/dyvm6xra/dyvm6xrauser45/fred/models--OpenOneRec--OneRec-8B-pro}"
MODEL_CONTAINER="${XLLM_MODEL_CONTAINER:-/models/OneRec-8B-pro}"
RECIF_HOST="${XLLM_RECIF_HOST:-/home/dyvm6xra/dyvm6xrauser45/fred/huggingface/hub/recif_bs_model/recif_beam_search}"
RECIF_CONTAINER="${XLLM_RECIF_CONTAINER:-/recif}"

if [[ ! -d "$MODEL_HOST" ]]; then
  echo "ERROR: model directory not found on host: $MODEL_HOST" >&2
  echo "Set XLLM_MODEL_HOST to the correct path, or download the model first." >&2
  exit 1
fi

RECIF_MOUNT=()
if [[ -d "$RECIF_HOST" ]]; then
  RECIF_MOUNT=(-v "${RECIF_HOST}:${RECIF_CONTAINER}:ro")
else
  echo "WARNING: recif directory not found on host: $RECIF_HOST" >&2
  echo "Set XLLM_RECIF_HOST if you need examples/bridge_moe_bs/fred_run.sh." >&2
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")"
  if [[ "$state" == "running" ]]; then
    echo "Container '$CONTAINER_NAME' is already running. Attach with:" >&2
    echo "  docker exec -it $CONTAINER_NAME bash" >&2
    exit 1
  fi
  echo "Removing existing container '$CONTAINER_NAME' (state: $state)..."
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -it \
  --gpus all \
  --privileged \
  --shm-size=128g \
  --ipc=host \
  --net=host \
  --pid=host \
  --name "$CONTAINER_NAME" \
  -e PYTHONPATH="/workspace/xllm/build/lib.linux-x86_64-cpython-312" \
  -v "${XLLM_DIR}:/workspace/xllm" \
  -v "${MODEL_HOST}:${MODEL_CONTAINER}:ro" \
  "${RECIF_MOUNT[@]}" \
  -w /workspace/xllm \
  "$IMAGE" \
  bash -c 'git config --global --add safe.directory "*" 2>/dev/null || true; exec /bin/bash'
