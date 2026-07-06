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

"""Torch Kineto trace helpers for offline xLLM REC benchmarks.

Controlled by ``TORCH_PROFILE_TRACES`` (default ``0``). When enabled, xLLM REC
runs drive the in-process C++ ``TorchProfiler`` (Chrome ``.pt.trace.json``),
mirroring vLLM-GR's ``VLLM_GR_PROFILE_TRACES`` workflow.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


def profile_traces_enabled() -> bool:
    value = os.environ.get("TORCH_PROFILE_TRACES", "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def resolve_profile_dir(out_dir: Optional[str]) -> str:
    override = os.environ.get("XLLM_PROFILE_DIR") or os.environ.get(
        "TORCH_PROFILE_DIR"
    )
    if override:
        return os.path.abspath(override)
    if out_dir:
        return os.path.abspath(os.path.join(out_dir, "traces"))
    return os.path.abspath("traces")


def configure_xllm_torch_profile(*, enabled: bool, profile_dir: str) -> None:
    import xllm_export

    configure = getattr(xllm_export, "configure_torch_profile", None)
    if not callable(configure):
        raise RuntimeError(
            "xllm_export.configure_torch_profile is missing; rebuild xllm_export."
        )
    configure(enabled, profile_dir)


def _start_profile(rec) -> bool:
    start = getattr(rec, "start_profile", None)
    if callable(start):
        return bool(start())
    master = getattr(rec, "master", None)
    if master is not None and callable(getattr(master, "start_profile", None)):
        return bool(master.start_profile())
    raise RuntimeError(
        "xLLM REC profiling is unavailable: rebuild xllm_export and sync "
        "xllm/pybind/rec.py into build/lib."
    )


def _stop_profile(rec) -> bool:
    stop = getattr(rec, "stop_profile", None)
    if callable(stop):
        return bool(stop())
    master = getattr(rec, "master", None)
    if master is not None and callable(getattr(master, "stop_profile", None)):
        return bool(master.stop_profile())
    raise RuntimeError(
        "xLLM REC profiling is unavailable: rebuild xllm_export and sync "
        "xllm/pybind/rec.py into build/lib."
    )


@contextmanager
def xllm_rec_profile_session(rec) -> Iterator[Optional[str]]:
    """Start/stop xLLM REC torch traces around a benchmark window."""
    if not profile_traces_enabled():
        print(
            "  (torch profile traces disabled — set TORCH_PROFILE_TRACES=1 to dump traces)"
        )
        yield None
        return

    profile_dir = resolve_profile_dir(getattr(rec, "_profile_dir", None))
    os.makedirs(profile_dir, exist_ok=True)
    print(f"[profile] torch traces enabled; writing to {profile_dir}")

    if not _start_profile(rec):
        raise RuntimeError("xLLM REC start_profile() failed")
    try:
        yield profile_dir
    finally:
        if not _stop_profile(rec):
            raise RuntimeError("xLLM REC stop_profile() failed")
        print(f"[profile] torch trace export complete under {profile_dir}")
