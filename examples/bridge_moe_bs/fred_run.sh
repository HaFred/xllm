#!/bin/bash
set -euo pipefail
clear
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XLLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_TAG="cpython-$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
PY_XLLM_PKG="${XLLM_ROOT}/build/lib.linux-x86_64-${PY_TAG}/xllm"
export PYTHONPATH="${XLLM_ROOT}/build/lib.linux-x86_64-${PY_TAG}${PYTHONPATH:+:${PYTHONPATH}}"

# Keep editable Python wrappers in sync with the repo (build/lib can lag behind).
if [[ -f "${XLLM_ROOT}/xllm/pybind/rec.py" && -d "${PY_XLLM_PKG}/pybind" ]]; then
  cp "${XLLM_ROOT}/xllm/pybind/rec.py" "${PY_XLLM_PKG}/pybind/rec.py"
fi

cd "${SCRIPT_DIR}"

# Mount your recif checkpoint into the container, e.g.:
#   -v /path/to/recif_beam_search:/recif
export RECIF_BACKEND="${RECIF_BACKEND:-xllm}"
RECIF_CKPT="${RECIF_CKPT:-/recif/checkpoint}"
RECIF_MODEL="${RECIF_MODEL:-${SCRIPT_DIR}/xllm_hf_export}"

TORCH_PROFILE_TRACES=1
# Set to 1 to dump torch .pt.trace.json files for Chrome/Perfetto analysis.
# Default 0: no trace files on disk (faster benchmark, no disk bloat).
export TORCH_PROFILE_TRACES="${TORCH_PROFILE_TRACES:-0}"

if [[ ! -f "${RECIF_MODEL}/model.safetensors" ]]; then
  echo "[info] exporting Megatron checkpoint to ${RECIF_MODEL}"
  python export_recif_to_hf.py \
    --ckpt "${RECIF_CKPT}" \
    --config "${RECIF_CKPT}/config.json" \
    --out-dir "${RECIF_MODEL}" \
    --fused-experts
fi

python recif_bs.py \
  --backend "${RECIF_BACKEND}" \
  --model "${RECIF_MODEL}" \
  --history "${RECIF_HISTORY:-598080194427,628177754964,755993681678}" \
  --bf "${RECIF_BF:-64,128,256}" \
  --beam "${RECIF_BEAM:-1000}" \
  --device "${RECIF_DEVICE:-cuda:1}" \
  --warmup "${RECIF_WARMUP:-10}" \
  --num-requests "${RECIF_NUM_REQUESTS:-64}" \
  --batch-size "${RECIF_BATCH_SIZE:-4}" \
  --max-decode-rounds "${RECIF_MAX_DECODE_ROUNDS:-3}" \
  --out-dir "${RECIF_OUT_DIR:-out-xllm/64requests-${RECIF_BACKEND}_profile}" \
  --just-log-one-output
