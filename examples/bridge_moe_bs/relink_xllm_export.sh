#!/usr/bin/env bash
# Incrementally recompile changed C++ sources and relink xllm_export.so without
# triggering CMake reconfigure (avoids yalantinglibs / Mooncake configure failures
# when ninja tries to regenerate build.ninja).
#
# Usage (inside xllm-cuda container):
#   cd /workspace/xllm/examples/bridge_moe_bs
#   ./relink_xllm_export.sh \
#     xllm/core/layers/common/attention_metadata_builder.cpp \
#     xllm/core/layers/cuda/xattention_planinfo.cpp
set -euo pipefail

XLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${XLLM_ROOT}/build/cmake.linux-x86_64-cpython-312"
CCDB="${BUILD_DIR}/compile_commands.json"
NINJA="${BUILD_DIR}/build.ninja"
EXPORT_SO="${XLLM_ROOT}/build/lib.linux-x86_64-cpython-312/xllm/xllm_export.cpython-312-x86_64-linux-gnu.so"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <repo-relative-source.cpp> [...]" >&2
  exit 1
fi

for path in "$@"; do
  if [[ ! -f "${XLLM_ROOT}/${path}" ]]; then
    echo "missing source: ${XLLM_ROOT}/${path}" >&2
    exit 1
  fi
done

if [[ ! -f "${CCDB}" || ! -f "${NINJA}" ]]; then
  echo "missing build tree under ${BUILD_DIR}; run a full pip install first" >&2
  exit 1
fi

ABS_SOURCES=()
for path in "$@"; do
  ABS_SOURCES+=("${XLLM_ROOT}/${path}")
done

echo "[relink] compiling ${#ABS_SOURCES[@]} source(s)"
python3 - "${CCDB}" "${BUILD_DIR}" "${XLLM_ROOT}" "${ABS_SOURCES[@]}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

ccdb, build_dir, root = sys.argv[1:4]
sources = sys.argv[4:]
build_dir = Path(build_dir)

with open(ccdb) as f:
    entries = {e["file"]: e for e in json.load(f)}

archives = {}
ninja_text = (build_dir / "build.ninja").read_text()

def archive_for_object(rel_obj: str) -> Path:
    needle = rel_obj.replace("\\", "/")
    for line in ninja_text.splitlines():
        if ".a:" not in line or needle not in line:
            continue
        archive_rel = line.split(":")[0].removeprefix("build ").strip()
        if archive_rel.endswith(".a"):
            return build_dir / archive_rel
    raise SystemExit(f"could not locate static archive for {rel_obj}")

for src in sources:
    entry = entries.get(src)
    if entry is None:
        raise SystemExit(f"compile_commands.json has no entry for {src}")
    print(f"  compile {Path(src).name}")
    subprocess.check_call(entry["command"], shell=True, cwd=entry["directory"])
    obj = None
    for candidate in Path(entry["directory"]).rglob(Path(src).name + ".o"):
        obj = candidate
        break
    if obj is None:
        raise SystemExit(f"could not find object file for {src}")
    rel_obj = str(obj.relative_to(build_dir)).replace("\\", "/")
    archive = archive_for_object(rel_obj)
    archives[str(archive)] = str(obj)

for archive, obj in archives.items():
    print(f"  ar update {archive}")
    subprocess.check_call(["ar", "rs", archive, obj])
    subprocess.check_call(["ranlib", archive])

# Rebuild each touched static archive with ninja's canonical ar rule so member
# order and completeness match a full build (incremental ar rs alone can leave
# stale objects when other translation units in the archive were rebuilt).
archive_targets = {
    "libruntime.a": "libruntime.a",
    "libcommon_layers.a": "libcommon_layers.a",
    "libcuda_layers.a": "libcuda_layers.a",
    "libcuda_kernels.a": "libcuda_kernels.a",
    "libblock.a": "libblock.a",
    "libscheduler.a": "libscheduler.a",
    "librequest.a": "librequest.a",
    "libbatch.a": "libbatch.a",
    "libdistributed_runtime.a": "libdistributed_runtime.a",
}
rebuilt = set()
for archive_path in archives:
    archive_name = Path(archive_path).name
    target = archive_targets.get(archive_name)
    if target is None or target in rebuilt:
        continue
    commands = subprocess.check_output(
        ["ninja", "-t", "commands", target],
        cwd=build_dir,
        text=True,
        stderr=subprocess.STDOUT,
    ).splitlines()
    ar_cmds = [line for line in commands if "/ar qc " in line or " ar qc " in line]
    if not ar_cmds:
        raise SystemExit(f"could not find ar qc command for target {target}")
    print(f"  rebuild archive target {target}")
    subprocess.check_call(ar_cmds[-1], shell=True, cwd=build_dir)
    rebuilt.add(target)
PY

echo "[relink] linking ${EXPORT_SO}"
python3 - "${BUILD_DIR}" "${EXPORT_SO}" <<'PY'
import subprocess
import sys
from pathlib import Path

build_dir = Path(sys.argv[1])
target = sys.argv[2]
commands = subprocess.check_output(
    ["ninja", "-t", "commands", Path(target).name],
    cwd=build_dir,
    text=True,
).splitlines()
link_cmds = [line for line in commands if line.startswith(": && /usr/bin/c++")]
if not link_cmds:
    raise SystemExit(f"could not find link command for {target}")
subprocess.check_call(link_cmds[-1], shell=True, cwd=build_dir)
PY

echo "[relink] done: ${EXPORT_SO}"
ls -la "${EXPORT_SO}"

# The editable xllm package loads xllm_export from ${XLLM_ROOT}/xllm/, not only
# from build/lib. Keep both locations in sync after incremental relinks.
cp -f "${EXPORT_SO}" "${XLLM_ROOT}/xllm/"
echo "[relink] synced $(basename "${EXPORT_SO}") -> ${XLLM_ROOT}/xllm/"
