#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XLLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_TAG="cpython-$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
export PYTHONPATH="${XLLM_ROOT}/build/lib.linux-x86_64-${PY_TAG}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${SCRIPT_DIR}"

XLLM_MAX_DECODE_ROUNDS=1

python run_rec_beam_search.py \
  --model "${XLLM_MODEL:-/models/OneRec-8B-pro}" \
  --devices "${XLLM_REC_DEVICE:-cuda:1}" \
  --beam-width "${XLLM_BEAM_WIDTH:-2}" \
  --max-decode-rounds "${XLLM_MAX_DECODE_ROUNDS:-1}" \
  --prompt "${XLLM_PROMPT:-where is bejing?}"
