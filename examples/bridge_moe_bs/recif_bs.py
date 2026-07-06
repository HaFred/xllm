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

"""recif hierarchical beam search runner for the xLLM bridge.

Default backend (``--backend hf``) runs the reference 3-head SID beam search
using the exported backbone + ``recif_external_heads.safetensors``. This matches
``recif_beam_search_standalone.py`` / ``recif_beam_search_concurrent.py``.

The optional ``--backend xllm`` path drives xLLM REC with recif external SID heads.

Usage:
  python examples/bridge_moe_bs/recif_bs.py \\
      --model ./xllm_hf_export \\
      --history 598080194427,628177754964,755993681678 \\
      --bf 64,128,256 --beam 1000 --device cuda:1 \\
      --num-requests 64 --batch-size 4 --warmup 10 \\
      --out-dir out-xllm/64requests --just-log-one-output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Sequence

from export_recif_to_hf import export_recif_checkpoint
from recif_common import (
    NUM_SID_LEVELS,
    VOCAB,
    build_prefix_tokens,
    parse_bf,
    parse_history,
    predictions_from_beam,
    resolve_output_path,
    write_json,
)
from recif_inference import beam_search, build_prefix_tensor, load_recif_model
from libs.mfu import (
    ModelPerfEstimator,
    RunPerfTotals,
    compute_run_mfu,
    default_peak_tflops,
    dtype_byte_size,
    mfu_to_dict,
    print_run_mfu,
    record_backbone_forward,
    record_external_head,
    record_recif_beam_search_batch,
)
from libs.perf_slo import (
    OUTPUT_TOKENS_PER_REQUEST,
    RequestPerf,
    calculate_metrics,
    mark_step,
    metrics_to_dict,
    print_stage_metrics,
    sync_device,
)

try:
    from xllm import BeamSearchParams, REC
except ImportError:
    BeamSearchParams = None  # type: ignore[misc, assignment]
    REC = None  # type: ignore[misc, assignment]


def _resolve_model_dir(args: argparse.Namespace) -> str:
    if args.model:
        if not os.path.isdir(args.model):
            raise FileNotFoundError(f"--model directory not found: {args.model}")
        return os.path.abspath(args.model)

    if not args.ckpt:
        raise ValueError("Provide --model (exported HF dir) or --ckpt + --config")

    config_path = args.config
    if not config_path:
        candidate = os.path.join(args.ckpt, "config.json")
        if os.path.isfile(candidate):
            config_path = candidate
    if not config_path or not os.path.isfile(config_path):
        raise FileNotFoundError("--config is required when --model is not set")

    if args.export_dir:
        export_dir = args.export_dir
    else:
        export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xllm_hf_export")
    if args.reexport or not os.path.isfile(os.path.join(export_dir, "model.safetensors")):
        print(f"[info] exporting Megatron checkpoint to {export_dir}")
        export_recif_checkpoint(
            args.ckpt,
            config_path,
            export_dir,
            fused_experts=args.fused_experts,
            validate=False,
        )
    return os.path.abspath(export_dir)


def _device_to_torch_device(device: str) -> str:
    if device.startswith("cuda:") or device == "cuda":
        return device if device != "cuda" else "cuda:0"
    raise ValueError(f"recif_bs.py currently expects a CUDA device, got {device!r}")


def _run_hf_request(
    model,
    heads,
    prefix,
    *,
    bf: tuple[int, int, int],
    beam: int,
    perf_totals: RunPerfTotals | None = None,
    perf_estimator: ModelPerfEstimator | None = None,
) -> tuple[List[dict], dict]:
    triples, scores, timings = beam_search(
        model,
        heads,
        prefix,
        bf=bf,
        beam=beam,
        return_timings=True,
        perf_totals=perf_totals,
        perf_estimator=perf_estimator,
    )
    return predictions_from_beam(triples, scores), timings


def _run_hf_batch(
    model,
    heads,
    prefix,
    *,
    bf: tuple[int, int, int],
    beam: int,
    batch_size: int,
    perf_totals: RunPerfTotals | None = None,
    perf_estimator: ModelPerfEstimator | None = None,
) -> tuple[List[List[dict]], List[dict]]:
    batch_predictions: List[List[dict]] = []
    batch_timings: List[dict] = []
    for _ in range(batch_size):
        predictions, timings = _run_hf_request(
            model,
            heads,
            prefix,
            bf=bf,
            beam=beam,
            perf_totals=perf_totals,
            perf_estimator=perf_estimator,
        )
        batch_predictions.append(predictions)
        batch_timings.append(timings)
    return batch_predictions, batch_timings


def _resolve_beam_and_topk(bf: tuple[int, int, int], beam: int) -> tuple[int, int]:
    bf_max = max(bf)
    top_k = max(bf_max, beam)
    if top_k % 8 != 0:
        top_k = ((top_k + 7) // 8) * 8
    return beam, top_k


def _sequence_to_prediction(seq) -> dict | None:
    from recif_common import decode_generated_tokens

    token_ids = list(getattr(seq, "token_ids", []) or [])
    decoded = decode_generated_tokens(token_ids)
    if decoded is None:
        return None
    sa, sb, sc = decoded
    log_prob = 0.0
    if getattr(seq, "logprobs", None):
        tail = seq.logprobs[-NUM_SID_LEVELS:]
        log_prob = float(sum(getattr(entry, "logprob", 0.0) for entry in tail))
    from recif_common import bytes_to_sid, sid_triple_to_token_ids

    return {
        "sid": [sa, sb, sc],
        "sid_int64": bytes_to_sid(sa, sb, sc),
        "log_prob": log_prob,
        # "token_ids": sid_triple_to_token_ids(sa, sb, sc),
    }


def _output_to_predictions(output, *, debug: bool = False) -> List[dict]:
    predictions = []
    outputs = getattr(output, "outputs", None) or []
    if debug:
        print(
            f"[debug] finish_reason={getattr(output, 'finish_reason', None)!r}, "
            f"num_outputs={len(outputs)}"
        )
    for seq_idx, seq in enumerate(outputs):
        token_ids = list(getattr(seq, "token_ids", []) or [])
        if debug:
            print(
                f"[debug] seq[{seq_idx}] index={getattr(seq, 'index', None)!r} "
                f"token_ids_len={len(token_ids)} tail={token_ids[-8:] if token_ids else []}"
            )
        row = _sequence_to_prediction(seq)
        if row is None:
            continue
        row["rank"] = int(getattr(seq, "index", len(predictions)))
        predictions.append(row)
    predictions.sort(key=lambda item: item["log_prob"], reverse=True)
    for rank, row in enumerate(predictions):
        row["rank"] = rank
    return predictions


# xLLM REC runs all decode rounds inside one generate() call and does not expose
# per-round timestamps. Allocate batch E2EL using HF-measured round shares.
_XLLM_ROUND_RATIOS = (0.138, 0.431, 0.431)


def _xllm_timings_from_e2el(e2el_s: float) -> dict:
    step_latencies = [e2el_s * ratio for ratio in _XLLM_ROUND_RATIOS]
    return {
        "step_latencies_s": step_latencies,
        "ttft_s": step_latencies[0],
        "itl_s": step_latencies[1:],
        "e2el_s": e2el_s,
    }


def _run_xllm_batch(
    rec,
    token_prompt: Sequence[int],
    params,
    batch_size: int,
    *,
    debug: bool = False,
) -> tuple[List[List[dict]], float]:
    prompts = [list(token_prompt) for _ in range(batch_size)]
    start = time.perf_counter()
    outputs = rec.beam_search_tokens(prompts, params=params)
    elapsed = time.perf_counter() - start
    if not outputs:
        raise RuntimeError("xLLM REC returned no outputs")
    if len(outputs) != batch_size:
        raise RuntimeError(
            f"xLLM REC returned {len(outputs)} outputs for batch_size={batch_size}"
        )
    return [_output_to_predictions(output, debug=debug) for output in outputs], elapsed


def _load_model_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="recif hierarchical beam search (HF reference or xLLM REC)",
    )
    parser.add_argument("--model", default="", help="Exported HF directory from export_recif_to_hf.py")
    parser.add_argument("--ckpt", default="", help="Megatron checkpoint directory (optional)")
    parser.add_argument("--config", default="", help="Base config.json for export")
    parser.add_argument("--export-dir", default="", help="Auto-export target when using --ckpt")
    parser.add_argument("--reexport", action="store_true", help="Force Megatron re-export")
    parser.add_argument("--fused-experts", action="store_true", help="Use stacked MoE export format")
    parser.add_argument(
        "--backend",
        choices=["hf", "xllm"],
        default="hf",
        help="hf = 3 external SID heads (correct); xllm = native REC (experimental)",
    )
    parser.add_argument("--history", default="", help="Comma-separated int64 SIDs")
    parser.add_argument("--bf", default="8,8,8", help="Branching factors per SID byte level")
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-decode-rounds", type=int, default=NUM_SID_LEVELS)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--max-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--out", default="preds.json")
    parser.add_argument("--perf-out", default="perf.json")
    parser.add_argument("--gpu-peak-tflops", type=float, default=None,
                        help="peak BF16/FP16 TFLOPS per GPU for MFU (default: auto)")
    parser.add_argument("--just-log-one-output", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    if args.backend == "xllm":
        if REC is None or BeamSearchParams is None:
            raise SystemExit(
                "xLLM backend requested but xllm is not importable. "
                "Set PYTHONPATH to the xLLM build directory or use --backend hf."
            )

    print("args:", args)

    bf = parse_bf(args.bf)
    history = parse_history(args.history)
    token_prompt = build_prefix_tokens(history)
    model_dir = _resolve_model_dir(args)

    if args.backend == "xllm":
        recif_meta = (_load_model_config(model_dir).get("recif") or {})
        if recif_meta:
            print(
                "[info] recif external SID heads will be loaded from xLLM REC "
                f"({recif_meta.get('external_heads_file', 'recif_external_heads.safetensors')})"
            )
        else:
            print(
                "[warn] model config has no recif metadata; xLLM REC will use lm_head only"
            )
    device = _device_to_torch_device(args.device)

    batch_size = max(1, args.batch_size)
    num_requests = max(1, args.num_requests)
    input_tokens = len(token_prompt)
    perf_out_path = resolve_output_path(args.out_dir, args.perf_out)
    out_path = resolve_output_path(args.out_dir, args.out) if args.out else ""
    base_cfg = _load_model_config(model_dir)
    peak_gpu_tflops = (
        args.gpu_peak_tflops
        if args.gpu_peak_tflops is not None
        else default_peak_tflops(device)
    )

    print(
        f"[info] backend={args.backend}, model={model_dir}, history={len(history)} items, "
        f"prefix_tokens={input_tokens}, bf={bf}, beam={args.beam}, "
        f"device={device}, warmup={args.warmup}, num_requests={num_requests}, "
        f"batch_size={batch_size}"
    )

    rec = None
    model = None
    heads = None
    prefix = None
    xllm_params = None
    perf_estimator = None

    import torch

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    perf_estimator = ModelPerfEstimator(
        base_cfg,
        weight_bytes=dtype_byte_size(dtype),
        activation_bytes=dtype_byte_size(dtype),
        external_vocab=VOCAB,
    )

    if args.backend == "hf":
        model, heads = load_recif_model(model_dir, device, dtype=dtype)
        prefix = build_prefix_tensor(history, device)
    else:
        beam_width, top_k = _resolve_beam_and_topk(bf, args.beam)
        rec = REC(
            model=model_dir,
            devices=device,
            beam_width=beam_width,
            max_decode_rounds=args.max_decode_rounds,
            block_size=args.block_size,
            max_tokens_per_batch=args.max_tokens_per_batch,
            max_seqs_per_batch=batch_size,
            enable_prefix_cache=False,
            enable_chunked_prefill=False,
        )
        xllm_params = BeamSearchParams(
            beam_width=beam_width,
            max_tokens=args.max_decode_rounds,
            top_k=top_k,
            top_logprobs=top_k,
            logprobs=True,
            num_return_sequences=beam_width,
        )

    for warmup_idx in range(max(0, args.warmup)):
        print(f"Warmup request {warmup_idx + 1} of {max(0, args.warmup)}")
        if args.backend == "hf":
            beam_search(model, heads, prefix, bf=bf, beam=args.beam)
        else:
            _run_xllm_batch(rec, token_prompt, xllm_params, batch_size=1, debug=args.debug)

    request_perfs: List[RequestPerf] = []
    run_perf_totals = RunPerfTotals()
    all_request_outputs: List[dict] = []
    bench_start = time.perf_counter()
    num_batches = (num_requests + batch_size - 1) // batch_size
    last_error: str | None = None
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_requests)
        current_batch_size = batch_end - batch_start
        batch_wall_start = time.perf_counter()
        try:
            if args.backend == "hf":
                batch_predictions, batch_timings = _run_hf_batch(
                    model,
                    heads,
                    prefix,
                    bf=bf,
                    beam=args.beam,
                    batch_size=current_batch_size,
                    perf_totals=run_perf_totals,
                    perf_estimator=perf_estimator,
                )
                sync_device(device)
                for offset in range(current_batch_size):
                    req_idx = batch_start + offset
                    timings = batch_timings[offset]
                    request_perfs.append(
                        RequestPerf(
                            success=True,
                            input_tokens=input_tokens,
                            output_tokens=OUTPUT_TOKENS_PER_REQUEST,
                            ttft_s=timings["ttft_s"],
                            e2el_s=timings["e2el_s"],
                            itl_s=list(timings["itl_s"]),
                        )
                    )
                    all_request_outputs.append(
                        {
                            "request_index": req_idx,
                            "batch_index": batch_idx,
                            "success": True,
                            "predictions": batch_predictions[offset],
                        }
                    )
            else:
                batch_predictions, batch_elapsed = _run_xllm_batch(
                    rec,
                    token_prompt,
                    xllm_params,
                    batch_size=current_batch_size,
                    debug=args.debug and batch_idx == 0,
                )
                sync_device(device)
                batch_timings = _xllm_timings_from_e2el(batch_elapsed)
                record_recif_beam_search_batch(
                    run_perf_totals,
                    perf_estimator,
                    batch_size=current_batch_size,
                    prefix_len=input_tokens,
                    bf=bf,
                    beam=args.beam,
                )
                for offset in range(current_batch_size):
                    req_idx = batch_start + offset
                    request_perfs.append(
                        RequestPerf(
                            success=True,
                            input_tokens=input_tokens,
                            output_tokens=OUTPUT_TOKENS_PER_REQUEST,
                            ttft_s=batch_timings["ttft_s"],
                            e2el_s=batch_elapsed,
                            itl_s=list(batch_timings["itl_s"]),
                        )
                    )
                    all_request_outputs.append(
                        {
                            "request_index": req_idx,
                            "batch_index": batch_idx,
                            "success": True,
                            "predictions": batch_predictions[offset],
                        }
                    )
        except Exception as exc:
            last_error = str(exc)
            sync_device(device)
            batch_e2el_s = time.perf_counter() - batch_wall_start
            for offset in range(current_batch_size):
                req_idx = batch_start + offset
                request_perfs.append(
                    RequestPerf(
                        success=False,
                        input_tokens=input_tokens,
                        output_tokens=0,
                        e2el_s=batch_e2el_s,
                        error=last_error,
                    )
                )
                all_request_outputs.append(
                    {
                        "request_index": req_idx,
                        "batch_index": batch_idx,
                        "success": False,
                        "error": last_error,
                        "predictions": [],
                    }
                )
    bench_duration = time.perf_counter() - bench_start

    successful_outputs = [row for row in all_request_outputs if row["success"]]
    if not successful_outputs:
        if rec is not None:
            rec.finish()
        raise RuntimeError(f"recif beam search failed: {last_error}")

    stage_name = "Beam Search" if args.backend == "hf" else "xLLM REC Beam Search"
    stage_metrics = calculate_metrics(request_perfs, bench_duration)
    print_stage_metrics(stage_metrics, stage_name=stage_name)

    run_mfu = compute_run_mfu(run_perf_totals, bench_duration, peak_gpu_tflops)
    if run_mfu is not None:
        print_run_mfu(run_mfu)
    else:
        print("[warn] MFU: could not be computed (zero duration or zero work)")

    stage = "recif_hf_beam_search" if args.backend == "hf" else "xllm_rec_beam_search"
    perf_payload = {
        "stage": stage,
        "metrics": metrics_to_dict(stage_metrics),
        "mfu": mfu_to_dict(run_mfu) if run_mfu is not None else None,
        "run_config": {
            "backend": args.backend,
            "model": model_dir,
            "history_items": len(history),
            "input_tokens": input_tokens,
            "bf": list(bf),
            "beam": args.beam,
            "device": device,
            "warmup": args.warmup,
            "num_requests": num_requests,
            "batch_size": batch_size,
            "max_decode_rounds": args.max_decode_rounds,
            "just_log_one_output": args.just_log_one_output,
            "output_tokens_per_request": OUTPUT_TOKENS_PER_REQUEST,
            "gpu_peak_tflops": peak_gpu_tflops,
        },
    }
    write_json(perf_out_path, perf_payload)
    print(f"[info] wrote performance metrics to {perf_out_path}")

    display_preds = (
        successful_outputs[0]["predictions"]
        if args.just_log_one_output
        else successful_outputs[-1]["predictions"]
    )
    print(f"\nTop-{len(display_preds)} predicted SID triples (sorted by log-prob):")
    print(f"{'rank':>4}  {'(sa, sb, sc)':>20}  {'sid_int64':>16}  {'log_prob':>10}")
    for row in display_preds:
        sa, sb, sc = row["sid"]
        print(
            f"{row['rank']:>4}  {f'({sa}, {sb}, {sc})':>20}  "
            f"{row['sid_int64']:>16}  {row['log_prob']:>10.4f}"
        )

    if out_path:
        if num_requests == 1 or args.just_log_one_output:
            out_payload = successful_outputs[0]["predictions"]
        else:
            out_payload = {
                "num_requests": num_requests,
                "batch_size": batch_size,
                "requests": all_request_outputs,
            }
        write_json(out_path, out_payload)
        total_preds = sum(len(row["predictions"]) for row in all_request_outputs)
        print(
            f"\n[info] wrote {len(all_request_outputs)} request(s), "
            f"{total_preds} total predictions to {out_path}"
        )

    if rec is not None:
        rec.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
