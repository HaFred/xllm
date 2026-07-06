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

"""Run REC xAttention + beam search on Qwen3-MoE (LlmRec path, CUDA).

Stack:
  REC -> RecMaster -> LlmRecMultiRoundPipeline -> Qwen3MoeForCausalLM
  -> XAttentionImpl (shared/unshared KV) -> rec_beam_search.cu

Requires xLLM built with USE_CUDA and `pip install -e .` so xllm_export loads.

Usage (resolve model from HF_HOME):
  python examples/bridge_moe/run_rec_beam_search_moe.py

Usage (explicit checkpoint):
  python examples/bridge_moe/run_rec_beam_search_moe.py \\
      --model /path/to/Qwen3-MoE-15B-A2B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hf_model_path import (  # noqa: E402
    DEFAULT_QWEN3_MOE_15B_REPO,
    assert_llmrec_model_type,
    resolve_model_path,
)

from xllm import BeamSearchParams, REC  # noqa: E402


def _parse_token_prompt(value: str) -> List[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def _print_output(index: int, output) -> None:
    print(f"=== request {index} ===")
    if output.prompt:
        print(f"prompt: {output.prompt!r}")
    if output.status is not None and not output.status.ok:
        print(f"status: {output.status.code} {output.status.message}")
        return
    for seq in output.outputs:
        print(f"beam {seq.index}: token_ids={list(seq.token_ids)}")
        if seq.text:
            print(f"  text={seq.text!r}")
        if seq.logprobs:
            print(f"  logprobs ({len(seq.logprobs)} entries):")
            for lp in seq.logprobs[:8]:
                print(f"    token_id={lp.token_id} logprob={lp.logprob}")
            if len(seq.logprobs) > 8:
                print(f"    ... {len(seq.logprobs) - 8} more")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="xLLM REC xAttention beam search for Qwen3-MoE"
    )
    parser.add_argument(
        "--model",
        default="",
        help="Local HF weights directory (default: newest snapshot under HF_HOME)",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_QWEN3_MOE_15B_REPO,
        help="Hugging Face repo id when --model is omitted",
    )
    parser.add_argument(
        "--hf-home",
        default="",
        help="Override HF_HOME for snapshot lookup",
    )
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=32,
        help="Beam width (lower if OOM on 15B MoE)",
    )
    parser.add_argument(
        "--max-decode-rounds",
        type=int,
        default=3,
        help="Must be >0 to enable LlmRecMultiRoundPipeline + xAttention",
    )
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--max-tokens-per-batch", type=int, default=16384)
    parser.add_argument("--max-seqs-per-batch", type=int, default=1)
    parser.add_argument("--max-memory-utilization", type=float, default=0.9)
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="where is bejing?",
        help="Text prompt (ignored when --token-prompt is set)",
    )
    parser.add_argument(
        "--token-prompt",
        default="",
        help="Comma-separated token ids",
    )
    parser.add_argument(
        "--config",
        default="",
        help="JSON file with BeamSearchParams fields",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use bundled smoke config (beam_width=8)",
    )
    args = parser.parse_args(argv)

    if args.max_decode_rounds <= 0:
        parser.error("--max-decode-rounds must be > 0 for REC xAttention")

    model_path = resolve_model_path(
        args.model or None,
        repo_id=args.hf_repo,
        hf_home=args.hf_home or None,
    )
    model_type = assert_llmrec_model_type(model_path)
    print(f"model_path: {model_path}")
    print(f"model_type: {model_type}")

    rec = REC(
        model=model_path,
        devices=args.devices,
        beam_width=args.beam_width,
        max_decode_rounds=args.max_decode_rounds,
        block_size=args.block_size,
        max_tokens_per_batch=args.max_tokens_per_batch,
        max_seqs_per_batch=args.max_seqs_per_batch,
        max_memory_utilization=args.max_memory_utilization,
        ep_size=args.ep_size,
        dp_size=args.dp_size,
        enable_prefix_cache=False,
        enable_chunked_prefill=False,
        enable_graph=False,
    )

    config_path = args.config
    if args.smoke and not config_path:
        config_path = str(_SCRIPT_DIR / "config_qwen3_moe_15b_a2b_smoke.json")

    beam_params_kwargs = {
        "beam_width": args.beam_width,
        "max_tokens": args.max_decode_rounds,
        "top_logprobs": args.beam_width,
        "logprobs": True,
    }
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            beam_params_kwargs.update(json.load(f))
    params = BeamSearchParams(**beam_params_kwargs)

    if args.token_prompt:
        token_prompts = [_parse_token_prompt(args.token_prompt)]
        outputs = rec.beam_search_tokens(token_prompts, params=params)
    else:
        outputs = rec.beam_search(args.prompt, params=params)

    for index, output in enumerate(outputs):
        _print_output(index, output)

    rec.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
