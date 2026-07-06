#!/usr/bin/env bash
set -euo pipefail

# View vLLM / PyTorch profiler traces in Perfetto from your laptop:
#   1) On GPU node:  bash scripts/vis_profile_results.sh
#   2) On laptop:    bash scripts/.forlocallaptop/remote_trace.sh <ssh-host>
#   3) Browser:       https://ui.perfetto.dev/  (accept "Use trace processor")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Which trace to load from each profile's torch/ dir: enginecore | apiserver
# Default enginecore (GPU worker). Overridden by TRACE_FILE if set.
VIEW_FILE=${VIEW_FILE:-enginecore}

# trace_processor speaks Perfetto's RPC protocol on this port (UI probes POST /status).
PORT=${PORT:-9001}
BIND_HOST=${BIND_HOST:-127.0.0.1}
SKIP_PORT_CLEAN=${SKIP_PORT_CLEAN:-1}
# trace_processor (recommended) | http (raw file download only) | auto
VIS_MODE=${VIS_MODE:-trace_processor}
TRACE_PROCESSOR=${TRACE_PROCESSOR:-"$REPO_DIR/.tools/trace_processor"}
TRACE_PROCESSOR_URL=${TRACE_PROCESSOR_URL:-https://get.perfetto.dev/trace_processor}

SSH_TARGET=${SSH_TARGET:-dgx}
REMOTE_USER=${REMOTE_USER:-$USER}
NODE_HOSTNAME=$(hostname -f 2>/dev/null || hostname)

DEFAULT_TRACE_DIR="$(python - "$REPO_DIR" <<'PY'
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
candidates = [p for p in repo.glob("profiles/*/torch") if p.is_dir()]
if not candidates:
    print("")
    sys.exit(0)
latest = max(candidates, key=lambda p: p.stat().st_mtime)
print(str(latest))
PY
)"

TRACE_PATH="${1:-$DEFAULT_TRACE_DIR}"
if [[ -n "${1:-}" ]]; then
  TRACE_SOURCE="explicit argument"
else
  TRACE_SOURCE="latest under $REPO_DIR/profiles/*/torch (by mtime)"
fi
# Support direct trace file path (e.g. /path/to/trace.pt.trace.json.gz)
if [[ -f "$TRACE_PATH" ]]; then
  TRACE_FILE="$(basename "$TRACE_PATH")"
  TRACE_DIR="$(cd "$(dirname "$TRACE_PATH")" && pwd)"
  PROFILE_ROOT="$(cd "$TRACE_DIR/../.." 2>/dev/null && pwd || echo "$TRACE_DIR")"
elif [[ -d "$TRACE_PATH/torch" ]]; then
  TRACE_DIR="$TRACE_PATH/torch"
  PROFILE_ROOT="$(cd "$TRACE_PATH" && pwd)"
  TRACE_FILE="${TRACE_FILE:-}"
else
  TRACE_DIR="$TRACE_PATH"
  if [[ "$(basename "$TRACE_DIR")" == "torch" ]]; then
    PROFILE_ROOT="$(cd "$(dirname "$TRACE_DIR")" && pwd)"
  else
    PROFILE_ROOT="$TRACE_DIR"
  fi
  TRACE_FILE="${TRACE_FILE:-}"
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [PROFILE_ROOT | TRACE_DIR | TRACE_FILE]

Serve PyTorch profiler traces for Perfetto (large traces need trace_processor).

PROFILE_ROOT   — a profile directory containing a torch/ subdirectory
TRACE_DIR      — the torch/ directory containing *.pt.trace.json.gz files
TRACE_FILE     — a direct path to a *.pt.trace.json.gz file

Env vars:
  VIEW_FILE           apiserver | enginecore — which trace to open (default: apiserver)
                      apiserver  → *async_llm* (APIServer CPU)
                      enginecore → *rank-0* (EngineCore GPU)
  VIS_MODE            trace_processor | http | auto (default: trace_processor)
  TRACE_FILE          Specific *.pt.trace.json.gz basename (overrides VIEW_FILE)
  PORT                Starting port; auto-increments if busy (default: 9001)
                      Multiple instances each get their own port.
  BIND_HOST           Bind address (default: 127.0.0.1)
  TRACE_PROCESSOR     Path to trace_processor binary
  SKIP_PORT_CLEAN     Set to 0 to kill old listeners on PORT before serving
  SSH_TARGET          SSH host alias for tunnel hint (default: dgx)

Workflow:
  Node:  bash scripts/vis_profile_results.sh
  Laptop: bash scripts/.forlocallaptop/remote_trace.sh <ssh-host>
  Browser: https://ui.perfetto.dev/  → click "YES, use loaded trace"

Do NOT use Perfetto "Open URL" deep-links for *.pt.trace.json.gz — they fail with tp-parse.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$TRACE_DIR" ]]; then
  echo "ERROR: no trace directory found under '$REPO_DIR/profiles/*/torch'." >&2
  echo "       Pass TRACE_DIR explicitly: $(basename "$0") /path/to/torch" >&2
  exit 1
fi

if [[ ! -d "$TRACE_DIR" ]]; then
  echo "ERROR: TRACE_DIR does not exist: $TRACE_DIR" >&2
  exit 1
fi

case "$VIEW_FILE" in
  apiserver|enginecore) ;;
  *)
    echo "ERROR: VIEW_FILE must be 'apiserver' or 'enginecore' (got: '$VIEW_FILE')." >&2
    exit 1
    ;;
esac

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is required but not found in PATH." >&2
  exit 1
fi

eval "$(python - "$TRACE_DIR" "$TRACE_FILE" "$VIEW_FILE" <<'PY'
import pathlib
import shlex
import subprocess
import sys

trace_dir = pathlib.Path(sys.argv[1])
requested = sys.argv[2].strip()
view_file = sys.argv[3].strip().lower()

patterns = ("*.pt.trace.json.gz", "*.json.gz", "*.json")
seen: set[pathlib.Path] = set()
files: list[pathlib.Path] = []
for pattern in patterns:
    for path in sorted(trace_dir.glob(pattern)):
        if path not in seen:
            seen.add(path)
            files.append(path)

valid: list[pathlib.Path] = []
corrupt: list[pathlib.Path] = []

for path in files:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        ok = subprocess.run(
            ["gzip", "-t", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if ok:
            valid.append(path)
        else:
            corrupt.append(path)
    else:
        valid.append(path)

selected: pathlib.Path | None = None
if requested:
    candidate = pathlib.Path(requested)
    if not candidate.is_absolute():
        candidate = trace_dir / requested
    if not candidate.exists():
        print('echo "ERROR: TRACE_FILE not found: {}" >&2'.format(candidate))
        print("exit 1")
        sys.exit(0)
    if candidate in corrupt:
        print('echo "ERROR: TRACE_FILE is corrupt (gzip -t failed): {}" >&2'.format(candidate))
        print("exit 1")
        sys.exit(0)
    selected = candidate
elif valid:
    if view_file == "apiserver":
        matches = [p for p in valid if "async_llm" in p.name]
        label = "*async_llm* (APIServer CPU)"
    else:
        matches = [p for p in valid if "rank-0" in p.name or "rank_0" in p.name]
        label = "*rank-0* (EngineCore GPU)"
    if matches:
        selected = max(matches, key=lambda p: p.stat().st_size)
    else:
        print(
            f'echo "ERROR: no valid {view_file} trace under {trace_dir} '
            f'(expected {label})" >&2',
        )
        print('echo "Available traces:" >&2')
        for path in valid:
            print(f'echo "  {path.name}" >&2')
        print("exit 1")
        sys.exit(0)

if corrupt:
    print(
        "Note: skipping corrupt trace(s): "
        + ", ".join(p.name for p in corrupt),
        file=sys.stderr,
    )

if selected:
    if requested:
        pick_reason = "TRACE_FILE"
    else:
        pick_reason = f"VIEW_FILE={view_file} ({label})"
    print(f"Viewing trace: {selected.name} ({pick_reason})", file=sys.stderr)
    others = [p for p in valid if p != selected]
    if others:
        print(
            "Other valid traces (VIEW_FILE=apiserver|enginecore or TRACE_FILE=<basename>):",
            file=sys.stderr,
        )
        for path in others:
            print(f"  {path.name}", file=sys.stderr)

if not selected:
    print('echo "ERROR: no valid trace files in {}" >&2'.format(trace_dir))
    print("exit 1")
    sys.exit(0)

print(f"SELECTED_TRACE={shlex.quote(str(selected))}")
PY
)"

if [[ -z "${SELECTED_TRACE:-}" ]]; then
  echo "ERROR: no valid trace files in $TRACE_DIR" >&2
  exit 1
fi

ensure_trace_processor() {
  if [[ -x "$TRACE_PROCESSOR" ]]; then
    return 0
  fi
  if command -v trace_processor >/dev/null 2>&1; then
    TRACE_PROCESSOR="$(command -v trace_processor)"
    return 0
  fi
  echo "Downloading trace_processor to $TRACE_PROCESSOR ..."
  mkdir -p "$(dirname "$TRACE_PROCESSOR")"
  curl -fsSL -o "$TRACE_PROCESSOR" "$TRACE_PROCESSOR_URL"
  chmod +x "$TRACE_PROCESSOR"
}

REQUESTED_PORT=$PORT
PORT="$(SKIP_PORT_CLEAN="$SKIP_PORT_CLEAN" python - "$PORT" "$BIND_HOST" <<'PY'
import os
import re
import signal
import socket
import subprocess
import sys
import time

requested = int(sys.argv[1])
host = sys.argv[2]
skip = os.environ.get("SKIP_PORT_CLEAN", "0") == "1"
max_tries = 30


def can_bind(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def collect_pids(port: int) -> set[int]:
    pids: set[int] = set()
    for cmd in (
        ["ss", "-lptn", f"sport = :{port}"],
        ["ss", "-tanp", f"sport = :{port}"],
    ):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            pids.update(int(m) for m in re.findall(r"pid=(\d+)", out))
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    try:
        out = subprocess.check_output(
            ["fuser", f"{port}/tcp"], text=True, stderr=subprocess.STDOUT
        )
        for tok in out.split():
            if tok.isdigit():
                pids.add(int(tok))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return pids


def free_port(port: int) -> None:
    if skip:
        return
    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    for pid in collect_pids(port):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(0.4)
    for pid in collect_pids(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(0.4)


if not skip:
    pids = collect_pids(requested)
    if pids:
        print(
            f"Freeing port {requested} (pids: {' '.join(map(str, sorted(pids)))}) ...",
            file=sys.stderr,
        )
    free_port(requested)
    # After killing, give the port a moment to release, then retry a few times
    for _ in range(6):
        if can_bind(requested):
            print(requested)
            sys.exit(0)
        time.sleep(0.5)
elif can_bind(requested):
    # Port is free and we're not killing anything — use it immediately
    print(requested)
    sys.exit(0)
else:
    # Port is busy but skip is set — don't wait, find next available immediately
    print(
        f"Port {requested} is busy; searching for next available port ...",
        file=sys.stderr,
    )

for offset in range(1, max_tries):
    port = requested + offset
    if can_bind(port):
        print(
            f"Using port {port} for this session.",
            file=sys.stderr,
        )
        print(port)
        sys.exit(0)

print(
    f"ERROR: no free port in [{requested}, {requested + max_tries - 1}] on {host}",
    file=sys.stderr,
)
sys.exit(1)
PY
)"

resolved_mode=$VIS_MODE
if [[ "$resolved_mode" == "auto" ]]; then
  resolved_mode=trace_processor
fi
if [[ "$resolved_mode" == "trace_processor" ]]; then
  if ! ensure_trace_processor; then
    echo "WARN: trace_processor unavailable; falling back to HTTP file server." >&2
    resolved_mode=http
  fi
fi

echo "Visualizing profile traces"
echo "  profile folder : $PROFILE_ROOT"
echo "  trace dir      : $TRACE_DIR"
echo "  selection      : $TRACE_SOURCE"
echo "  view target    : $VIEW_FILE${TRACE_FILE:+" (TRACE_FILE overrides)"}"
echo
echo "Trace viewer config"
echo "  mode           : $resolved_mode"
echo "  selected trace : $SELECTED_TRACE"
echo "  bind host      : $BIND_HOST"
echo "  port           : $PORT"
if [[ "$PORT" != "$REQUESTED_PORT" ]]; then
  echo "  (requested     : $REQUESTED_PORT — busy, auto-selected $PORT)"
fi
echo "  node hostname  : $NODE_HOSTNAME"
echo

echo "Step 1 (GPU node): keep this process running."
echo
echo "Step 2 (laptop): SSH tunnel"
if [[ -n "$SSH_TARGET" ]]; then
  echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} ${REMOTE_USER}@${SSH_TARGET}"
else
  echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} ${REMOTE_USER}@<your-ssh-host>"
fi
echo
echo "Step 3 (laptop browser):"
echo
echo "  One-time setup (only needed once per browser for non-9001 ports):"
echo "    Open https://ui.perfetto.dev/#!/flags/cspAllowAnyWebsocketPort"
echo "    and enable the \"Relax CSP\" flag."
echo "    (If the flag is not visible, try the Canary channel:"
echo "     https://ui.perfetto.dev/#!/flags/cspAllowAnyWebsocketPort?channel=canary)"
echo
if [[ "$resolved_mode" == "trace_processor" ]]; then
  if [[ "$PORT" == "9001" ]]; then
    echo "  Open: https://ui.perfetto.dev/"
    echo "  → Perfetto probes port 9001 by default and auto-detects the trace processor."
  else
    echo "  Open: https://ui.perfetto.dev/#!/?rpc_port=${PORT}"
    echo "  → Tells Perfetto to connect to port ${PORT} instead of the default 9001."
  fi
  echo "  → Click \"YES, use loaded trace\" (first load of a 100MB+ trace can take several minutes)."
  echo
  echo "  Serving: $(basename "$SELECTED_TRACE")"
  echo "  (PyTorch Chrome JSON — do NOT use Perfetto URL/deep-link import; it fails with tp-parse.)"
else
  echo "  → Download trace: curl -O http://127.0.0.1:${PORT}/$(basename "$SELECTED_TRACE")"
  echo "  → In Perfetto: \"Open trace file\" and pick the downloaded .gz"
fi
echo

if [[ "$resolved_mode" == "trace_processor" ]]; then
  echo "Starting trace_processor (Ctrl-C to stop) ..."
  echo "Loading trace into memory; please wait ..."
  exec "$TRACE_PROCESSOR" \
    --httpd \
    --http-port "$PORT" \
    --http-ip-address "$BIND_HOST" \
    "$SELECTED_TRACE"
fi

echo "Starting HTTP file server (download only; Ctrl-C to stop) ..."

cd "$TRACE_DIR"
exec python - "$PORT" "$BIND_HOST" <<'PY'
import http.server
import signal
import socketserver
import sys

port = int(sys.argv[1])
bind_host = sys.argv[2]


class PerfettoCorsHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "https://ui.perfetto.dev")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


class ReuseAddrTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


httpd = ReuseAddrTCPServer((bind_host, port), PerfettoCorsHandler)


def _shutdown(signum, frame):
    httpd.shutdown()


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

print(f"Serving files on http://{bind_host}:{port}", flush=True)
try:
    httpd.serve_forever()
finally:
    httpd.server_close()
PY
