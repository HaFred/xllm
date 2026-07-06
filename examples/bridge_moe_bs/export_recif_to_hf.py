#!/usr/bin/env python3
# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Export a recif Megatron checkpoint to an xLLM-friendly HF directory layout.

Input (rank-0 / single-EP):
  <ckpt>/_model_rank0.pt
  <ckpt>/external_rank0.pt
  <config_json>

Output:
  <out_dir>/config.json
  <out_dir>/model.safetensors                 # Qwen3-MoE backbone only
  <out_dir>/recif_external_heads.safetensors  # 3 external SID heads
  <out_dir>/recif_export_meta.json

Usage:
  python examples/bridge_moe_bs/export_recif_to_hf.py \\
      --ckpt /path/to/checkpoint \\
      --config /path/to/checkpoint/config.json \\
      --out-dir /path/to/recif_hf_export
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch
from safetensors.torch import save_file

from recif_common import (
    RECIF_CONFIG_KEY,
    build_recif_config,
    copy_tokenizer_files,
    load_arch_from_config,
    load_external_heads_state_dict,
    load_megatron_backbone_state_dict,
    megatron_to_hf,
)


def _save_safetensors(path: str, tensors: Dict[str, torch.Tensor]) -> None:
    cpu_tensors = {key: value.contiguous().cpu() for key, value in tensors.items()}
    save_file(cpu_tensors, path)


def _validate_with_transformers(out_dir: str) -> None:
    from transformers import Qwen3MoeConfig, Qwen3MoeModel

    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config_dict = json.load(handle)
    config = Qwen3MoeConfig(**config_dict)
    model = Qwen3MoeModel(config)
    from safetensors.torch import load_file

    model_path = os.path.join(out_dir, "model.safetensors")
    state_dict = load_file(model_path)
    stripped = {
        (key[len("model.") :] if key.startswith("model.") else key): value
        for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    real_missing = [
        name for name in missing if "rotary" not in name and "inv_freq" not in name
    ]
    if real_missing:
        raise RuntimeError(f"transformers validation missing keys: {real_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"transformers validation unexpected keys: {unexpected[:10]}")


def export_recif_checkpoint(
    ckpt_dir: str,
    config_path: str,
    out_dir: str,
    *,
    fused_experts: bool = False,
    validate: bool = False,
    tokenizer_dir: str | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    with open(config_path, encoding="utf-8") as handle:
        base_config = json.load(handle)
    arch = load_arch_from_config(config_path)

    megatron_sd = load_megatron_backbone_state_dict(ckpt_dir)
    hf_sd = megatron_to_hf(megatron_sd, arch, fused_experts=fused_experts)
    external_heads = load_external_heads_state_dict(ckpt_dir)

    config = build_recif_config(base_config)
    config_path_out = os.path.join(out_dir, "config.json")
    with open(config_path_out, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    model_path = os.path.join(out_dir, "model.safetensors")
    heads_path = os.path.join(out_dir, config[RECIF_CONFIG_KEY]["external_heads_file"])
    _save_safetensors(model_path, hf_sd)
    _save_safetensors(heads_path, external_heads)

    meta = {
        "source_ckpt": os.path.abspath(ckpt_dir),
        "source_config": os.path.abspath(config_path),
        "fused_experts": fused_experts,
        "num_backbone_tensors": len(hf_sd),
        "num_external_head_tensors": len(external_heads),
        "backbone_file": os.path.basename(model_path),
        "external_heads_file": os.path.basename(heads_path),
    }
    meta_path = os.path.join(out_dir, "recif_export_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    if tokenizer_dir:
        copied = copy_tokenizer_files(tokenizer_dir, out_dir)
        meta["tokenizer_dir"] = os.path.abspath(tokenizer_dir)
        meta["tokenizer_files"] = copied
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    if validate:
        _validate_with_transformers(out_dir)

    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export recif Megatron checkpoint to HF layout")
    parser.add_argument("--ckpt", required=True, help="Megatron checkpoint directory")
    parser.add_argument("--config", required=True, help="Base HF config.json")
    parser.add_argument("--out-dir", required=True, help="Output HF directory")
    parser.add_argument(
        "--fused-experts",
        action="store_true",
        help="Export MoE experts as stacked gate_up_proj/down_proj tensors",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Load exported backbone with transformers.Qwen3MoeModel",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=os.environ.get("RECIF_TOKENIZER_DIR", "/models/OneRec-8B-pro"),
        help="HF model dir to copy tokenizer.json from (xLLM init only)",
    )
    args = parser.parse_args(argv)

    out_dir = export_recif_checkpoint(
        args.ckpt,
        args.config,
        args.out_dir,
        fused_experts=args.fused_experts,
        validate=args.validate,
        tokenizer_dir=args.tokenizer_dir,
    )
    print(f"[ok] exported recif checkpoint to {out_dir}")
    print("[info] external SID heads are in recif_external_heads.safetensors")
    print("[info] xLLM REC still needs native 3-head recif support before inference parity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
