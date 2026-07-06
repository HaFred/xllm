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

"""Resolve Hugging Face hub snapshots under HF_HOME."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_QWEN3_MOE_15B_REPO = "Qwen/Qwen3-MoE-15B-A2B"


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))


def _repo_cache_dir(repo_id: str, hf_home: Optional[Path] = None) -> Path:
    root = hf_home or _hf_home()
    return root / "hub" / f"models--{repo_id.replace('/', '--')}"


def _newest_snapshot(snapshots_dir: Path) -> Optional[Path]:
    if not snapshots_dir.is_dir():
        return None
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _has_config(path: Path) -> bool:
    return (path / "config.json").is_file()


def resolve_hf_snapshot(
    repo_id: str,
    *,
    hf_home: Optional[str] = None,
) -> str:
    """Return the newest local snapshot path for a Hugging Face repo id."""
    home = Path(hf_home) if hf_home else _hf_home()
    cache_dir = _repo_cache_dir(repo_id, home)
    snapshot = _newest_snapshot(cache_dir / "snapshots")
    if snapshot is not None and _has_config(snapshot):
        return str(snapshot)

    raise FileNotFoundError(
        f"No local snapshot for {repo_id!r} under {cache_dir}. "
        "Download first, e.g.:\n"
        f"  huggingface-cli download {repo_id}"
    )


def resolve_model_path(
    model: Optional[str] = None,
    *,
    repo_id: str = DEFAULT_QWEN3_MOE_15B_REPO,
    hf_home: Optional[str] = None,
) -> str:
    """Resolve an explicit path or a HF hub snapshot."""
    if model:
        path = Path(os.path.expanduser(model)).resolve()
        if not _has_config(path):
            raise FileNotFoundError(f"Missing config.json under {path}")
        return str(path)
    return resolve_hf_snapshot(repo_id, hf_home=hf_home)


def read_model_type(model_path: str) -> str:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        data = json.load(f)
    model_type = data.get("model_type") or data.get("model_name")
    if not model_type:
        raise ValueError(f"config.json under {model_path} has no model_type")
    return str(model_type)


def assert_llmrec_model_type(model_path: str) -> str:
    model_type = read_model_type(model_path)
    if model_type not in {"qwen2", "qwen3", "qwen3_moe"}:
        raise ValueError(
            f"REC LlmRec xAttention expects qwen2/qwen3/qwen3_moe, got {model_type!r}"
        )
    return model_type
