#!/usr/bin/env bash
# Copyright 2025-2026 The xLLM Authors.
#
# Two-GPU expert-parallel REC xAttention for Qwen3-MoE-15B-A2B.
#
# Example:
#   XLLM_MODEL=/path/to/Qwen3-MoE-15B-A2B \
#   bash examples/bridge_moe/launch_rec_beam_search_multi_gpu.sh
#
# Override GPU list:
#   XLLM_REC_DEVICES=cuda:0,cuda:1 XLLM_EP_SIZE=2 ...

set -euo pipefail

: "${XLLM_REC_DEVICES:=cuda:0,cuda:1}"
: "${XLLM_EP_SIZE:=2}"
: "${XLLM_BEAM_WIDTH:=32}"
: "${XLLM_MAX_DECODE_ROUNDS:=3}"

export XLLM_REC_DEVICES
export XLLM_EP_SIZE
export XLLM_BEAM_WIDTH
export XLLM_MAX_DECODE_ROUNDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/launch_rec_beam_search.sh"
