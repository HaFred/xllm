#!/bin/bash
# Send beam-search requests to the xLLM HTTP server started by run_serving_in_cont.sh.
#
# Usage (from inside the container, while the server is running):
#   bash run_bs_requests.sh
#
# From the host:
#   docker exec -it xllm-cuda bash /workspace/xllm/run_bs_requests.sh

set -euo pipefail
clear
XLLM_TOP_LOGPROBS=2 
XLLM_BEAM_WIDTH=2
export XLLM_SERVER_HOST="${XLLM_SERVER_HOST:-127.0.0.1}"
export XLLM_SERVER_PORT="${XLLM_SERVER_PORT:-18000}"
export XLLM_MODEL_NAME="${XLLM_MODEL_NAME:-OneRec-8B-pro}"
export XLLM_BEAM_WIDTH="${XLLM_BEAM_WIDTH:-2}"
export XLLM_TOP_LOGPROBS="${XLLM_TOP_LOGPROBS:-4}"
export XLLM_MAX_TOKENS="${XLLM_MAX_TOKENS:-20}"
export XLLM_NUM_RETURN_SEQUENCES="${XLLM_NUM_RETURN_SEQUENCES:-$XLLM_BEAM_WIDTH}"

python - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

host = os.environ["XLLM_SERVER_HOST"]
port = os.environ["XLLM_SERVER_PORT"]
base_url = f"http://{host}:{port}"
model = os.environ["XLLM_MODEL_NAME"]
beam_width = int(os.environ["XLLM_BEAM_WIDTH"])
top_logprobs = int(os.environ["XLLM_TOP_LOGPROBS"])
max_tokens = int(os.environ["XLLM_MAX_TOKENS"])
num_return_sequences = int(os.environ["XLLM_NUM_RETURN_SEQUENCES"])

prompts = [
    "Hello, my name is ",
    "The president of the United States is ",
    "The capital of France is ",
    "The future of AI is ",
]

def check_server() -> None:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(f"{base_url}{path}", timeout=5):
                return
        except urllib.error.HTTPError:
            return
        except Exception:
            continue
    print(f"ERROR: xLLM server not reachable at {base_url}", file=sys.stderr)
    print("Start it first with: bash run_serving_in_cont.sh", file=sys.stderr)
    sys.exit(1)

def post_completion(prompt: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "beam_width": beam_width,
        "logprobs": top_logprobs,
        "num_return_sequences": num_return_sequences,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)

def sequence_logprob_sum(choice: dict) -> float | None:
    logprobs = choice.get("logprobs") or {}
    token_logprobs = logprobs.get("token_logprobs")
    if not token_logprobs:
        return None
    return float(sum(token_logprobs))

def print_beam_results(data: dict) -> None:
    choices = data.get("choices", [])
    if not choices:
        print("  (no choices returned)")
        return

    print(f"  returned {len(choices)} beam sequence(s):")
    for beam_idx, choice in enumerate(choices):
        text = choice.get("text", "")
        finish_reason = choice.get("finish_reason", "")
        logprob_sum = sequence_logprob_sum(choice)
        header = f"  [beam {beam_idx}]"
        if logprob_sum is not None:
            header += f" logprob_sum={logprob_sum:.4f}"
        if finish_reason:
            header += f" finish_reason={finish_reason!r}"
        print(header)
        print(f"    text: {text!r}")

check_server()
print(f"Sending {len(prompts)} beam-search requests to {base_url}/v1/completions")
print(
    f"  model={model} beam_width={beam_width} "
    f"num_return_sequences={num_return_sequences} "
    f"logprobs={top_logprobs} max_tokens={max_tokens}\n"
)

for prompt in prompts:
    print(f"Prompt: {prompt!r}")
    data = post_completion(prompt)
    print_beam_results(data)
    print()
PY
