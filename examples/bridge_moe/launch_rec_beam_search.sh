#!/usr/bin/env bash
# Copyright 2025-2026 The xLLM Authors.
#
# Launch REC xAttention beam search for Qwen3-MoE-15B-A2B from HF_HOME.
#
# Prerequisites:
#   - xLLM built with CUDA: pip install -e .
#   - Weights cached locally, e.g.:
#       huggingface-cli download Qwen/Qwen3-MoE-15B-A2B
#
# Environment overrides:
#   XLLM_MODEL          explicit checkpoint directory
#   HF_HOME             Hugging Face cache root (default ~/.cache/huggingface)
#   XLLM_REC_DEVICES    default cuda:0
#   XLLM_BEAM_WIDTH     default 32
#   XLLM_MAX_DECODE_ROUNDS default 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

: "${XLLM_REC_DEVICES:=cuda:0}"
: "${XLLM_BEAM_WIDTH:=32}"
: "${XLLM_MAX_DECODE_ROUNDS:=3}"
: "${XLLM_MAX_SEQS_PER_BATCH:=1}"
: "${XLLM_MAX_TOKENS_PER_BATCH:=16384}"
: "${XLLM_MAX_MEMORY_UTILIZATION:=0.9}"
: "${XLLM_EP_SIZE:=1}"
: "${XLLM_PROMPT:=where is bejing?}"

ARGS=(
  examples/bridge_moe/run_rec_beam_search_moe.py
  --devices "${XLLM_REC_DEVICES}"
  --beam-width "${XLLM_BEAM_WIDTH}"
  --max-decode-rounds "${XLLM_MAX_DECODE_ROUNDS}"
  --max-seqs-per-batch "${XLLM_MAX_SEQS_PER_BATCH}"
  --max-tokens-per-batch "${XLLM_MAX_TOKENS_PER_BATCH}"
  --max-memory-utilization "${XLLM_MAX_MEMORY_UTILIZATION}"
  --ep-size "${XLLM_EP_SIZE}"
  --prompt "${XLLM_PROMPT}"
  --config examples/bridge_moe/config_qwen3_moe_15b_a2b.json
)

if [[ -n "${XLLM_MODEL:-}" ]]; then
  ARGS+=(--model "${XLLM_MODEL}")
fi

if [[ -n "${XLLM_HF_REPO:-}" ]]; then
  ARGS+=(--hf-repo "${XLLM_HF_REPO}")
fi

if [[ -n "${XLLM_HF_HOME:-}" ]]; then
  ARGS+=(--hf-home "${XLLM_HF_HOME}")
fi

if [[ "${XLLM_SMOKE:-0}" == "1" ]]; then
  ARGS+=(--smoke)
fi

echo "HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}"
echo "Running: python ${ARGS[*]}"
exec python "${ARGS[@]}"
