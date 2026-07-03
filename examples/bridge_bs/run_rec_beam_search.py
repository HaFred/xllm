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

"""Run native xLLM REC beam search (backend=rec) on CUDA.

This example uses the full RecMaster stack:
  FixedStepsScheduler -> RecEngine -> LlmRecMultiRoundPipeline
  -> rec_beam_search.cu + xAttention shared/unshared KV.

It does NOT call HuggingFace Transformers for model forward.

Usage:
  python examples/bridge_bs/run_rec_beam_search.py \\
      --model /path/to/Qwen3-0.6B \\
      --devices cuda:0

  # Pre-tokenized prompt (matches xllm/c_api/examples/test_query_rec_completions.cpp):
  python examples/bridge_bs/run_rec_beam_search.py \\
      --model /path/to/Qwen3-0.6B \\
      --token-prompt 2870,374,387,98168,30
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from xllm import BeamSearchParams, REC


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
    parser = argparse.ArgumentParser(description="xLLM REC beam search bridge")
    parser.add_argument("--model", required=True, help="HF weights directory")
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-decode-rounds", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--max-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--max-seqs-per-batch", type=int, default=4)
    parser.add_argument(
        "--prompt",
        default="where is bejing?",
        help="Text prompt for LlmRec (ignored when --token-prompt is set)",
    )
    parser.add_argument(
        "--token-prompt",
        default="",
        help="Comma-separated token ids (Qwen3 tokenizer ids for the C API example)",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional JSON file with BeamSearchParams fields",
    )
    args = parser.parse_args(argv)

    rec = REC(
        model=args.model,
        devices=args.devices,
        beam_width=args.beam_width,
        max_decode_rounds=args.max_decode_rounds,
        block_size=args.block_size,
        max_tokens_per_batch=args.max_tokens_per_batch,
        max_seqs_per_batch=args.max_seqs_per_batch,
        enable_prefix_cache=False,
        enable_chunked_prefill=False,
    )

    beam_params_kwargs = {
        "beam_width": args.beam_width,
        "max_tokens": args.max_decode_rounds,
        "top_logprobs": args.beam_width,
        "logprobs": True,
    }
    if args.config:
        with open(args.config, encoding="utf-8") as f:
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
