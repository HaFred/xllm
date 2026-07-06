#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Host-mounted repos are owned by a non-root user; root in Docker hits "dubious ownership".
git config --global --add safe.directory '*'
git config --global --add safe.directory "$SCRIPT_DIR"

git submodule update --init --recursive
pip install pre-commit
pre-commit install || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python setup.py build --device cuda
