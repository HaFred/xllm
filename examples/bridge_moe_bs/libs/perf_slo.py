"""Stage latency/throughput metrics (bench.py-compatible SLO reporting)."""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import torch

DEFAULT_PERCENTILES = [50.0, 90.0, 95.0, 99.0]
OUTPUT_TOKENS_PER_REQUEST = 3  # three byte-level SID autoregressive steps


@dataclass
class RequestPerf:
    success: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    e2el_s: float = 0.0
    itl_s: list[float] = field(default_factory=list)
    error: str = ""


@dataclass
class StageMetrics:
    completed: int = 0
    failed: int = 0
    total_input: int = 0
    total_output: int = 0
    duration: float = 0.0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    total_token_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0
    percentiles_ttft_ms: list[tuple[float, float]] = field(default_factory=list)
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0
    percentiles_tpot_ms: list[tuple[float, float]] = field(default_factory=list)
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    percentiles_itl_ms: list[tuple[float, float]] = field(default_factory=list)
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    std_e2el_ms: float = 0.0
    percentiles_e2el_ms: list[tuple[float, float]] = field(default_factory=list)


def sync_device(device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def mark_step(device, start_t: float, step_latencies: list[float]) -> None:
    sync_device(device)
    step_latencies.append(time.perf_counter() - start_t)


def _percentile_stats(values, percentiles):
    if not values:
        zeros = [0.0] * len(percentiles)
        return 0.0, 0.0, 0.0, list(zip(percentiles, zeros))
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.std(arr)),
        [(p, float(np.percentile(arr, p))) for p in percentiles],
    )


def calculate_metrics(
    outputs: list[RequestPerf],
    dur_s: float,
    selected_percentiles: list[float] | None = None,
) -> StageMetrics:
    """Aggregate per-request timings into stage metrics (bench.py compatible)."""
    selected_percentiles = selected_percentiles or DEFAULT_PERCENTILES
    completed = 0
    failed = 0
    total_input = 0
    total_output = 0
    ttfts: list[float] = []
    tpots: list[float] = []
    itls: list[float] = []
    e2els: list[float] = []

    for out in outputs:
        if out.success:
            completed += 1
            total_input += out.input_tokens
            total_output += out.output_tokens
            ttfts.append(out.ttft_s)
            e2els.append(out.e2el_s)
            itls.extend(out.itl_s)
            if out.output_tokens > 1:
                tpots.append((out.e2el_s - out.ttft_s) / (out.output_tokens - 1))
        else:
            failed += 1

    mean_ttft, median_ttft, std_ttft, pct_ttft = _percentile_stats(ttfts, selected_percentiles)
    mean_tpot, median_tpot, std_tpot, pct_tpot = _percentile_stats(tpots, selected_percentiles)
    mean_itl, median_itl, std_itl, pct_itl = _percentile_stats(itls, selected_percentiles)
    mean_e2el, median_e2el, std_e2el, pct_e2el = _percentile_stats(e2els, selected_percentiles)

    return StageMetrics(
        completed=completed,
        failed=failed,
        total_input=total_input,
        total_output=total_output,
        duration=dur_s,
        request_throughput=completed / dur_s if dur_s > 0 else 0.0,
        output_throughput=total_output / dur_s if dur_s > 0 else 0.0,
        total_token_throughput=(total_input + total_output) / dur_s if dur_s > 0 else 0.0,
        mean_ttft_ms=mean_ttft * 1000,
        median_ttft_ms=median_ttft * 1000,
        std_ttft_ms=std_ttft * 1000,
        percentiles_ttft_ms=[(p, v * 1000) for p, v in pct_ttft],
        mean_tpot_ms=mean_tpot * 1000,
        median_tpot_ms=median_tpot * 1000,
        std_tpot_ms=std_tpot * 1000,
        percentiles_tpot_ms=[(p, v * 1000) for p, v in pct_tpot],
        mean_itl_ms=mean_itl * 1000,
        median_itl_ms=median_itl * 1000,
        std_itl_ms=std_itl * 1000,
        percentiles_itl_ms=[(p, v * 1000) for p, v in pct_itl],
        mean_e2el_ms=mean_e2el * 1000,
        median_e2el_ms=median_e2el * 1000,
        std_e2el_ms=std_e2el * 1000,
        percentiles_e2el_ms=[(p, v * 1000) for p, v in pct_e2el],
    )


def print_stage_metrics(m: StageMetrics, stage_name: str = "Beam Search") -> None:
    """Print stage performance metrics in the same layout as openonerec/bench.py."""
    print("{s:{c}^{n}}".format(s=f" {stage_name} Benchmark Result ", n=60, c="="))
    print("{:<45} {:<10}".format("Successful requests:", m.completed))
    print("{:<45} {:<10}".format("Failed requests:", m.failed))
    print("{:<45} {:<10.2f}".format("Benchmark duration (s):", m.duration))
    print("{:<45} {:<10}".format("Total input tokens:", m.total_input))
    print("{:<45} {:<10}".format("Total generated tokens:", m.total_output))
    print("{:<45} {:<10.2f}".format("Request throughput (req/s):", m.request_throughput))
    print("{:<45} {:<10.2f}".format("Output token throughput (tok/s):", m.output_throughput))
    print("{:<45} {:<10.2f}".format("Total token throughput (tok/s):", m.total_token_throughput))

    def _print_one(attr: str, name: str, header: str):
        print("{s:{c}^{n}}".format(s=header, n=60, c="-"))
        print("{:<45} {:<10.2f}".format(f"Mean {name} (ms):", getattr(m, f"mean_{attr}_ms")))
        print("{:<45} {:<10.2f}".format(f"Median {name} (ms):", getattr(m, f"median_{attr}_ms")))
        for p, value in getattr(m, f"percentiles_{attr}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<45} {:<10.2f}".format(f"P{p_word} {name} (ms):", value))

    _print_one("ttft", "TTFT", "Time to First Token")
    _print_one("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    _print_one("itl", "ITL", "Inter-token Latency")
    _print_one("e2el", "E2EL", "End-to-end Latency")
    print("=" * 60)


def metrics_to_dict(m: StageMetrics) -> dict:
    def _pairs(pairs):
        return [[p, v] for p, v in pairs]

    return {
        "duration": m.duration,
        "completed": m.completed,
        "failed": m.failed,
        "total_input_tokens": m.total_input,
        "total_output_tokens": m.total_output,
        "request_throughput": m.request_throughput,
        "output_throughput": m.output_throughput,
        "total_token_throughput": m.total_token_throughput,
        "mean_ttft_ms": m.mean_ttft_ms,
        "median_ttft_ms": m.median_ttft_ms,
        "std_ttft_ms": m.std_ttft_ms,
        "percentiles_ttft_ms": _pairs(m.percentiles_ttft_ms),
        "mean_tpot_ms": m.mean_tpot_ms,
        "median_tpot_ms": m.median_tpot_ms,
        "std_tpot_ms": m.std_tpot_ms,
        "percentiles_tpot_ms": _pairs(m.percentiles_tpot_ms),
        "mean_itl_ms": m.mean_itl_ms,
        "median_itl_ms": m.median_itl_ms,
        "std_itl_ms": m.std_itl_ms,
        "percentiles_itl_ms": _pairs(m.percentiles_itl_ms),
        "mean_e2el_ms": m.mean_e2el_ms,
        "median_e2el_ms": m.median_e2el_ms,
        "std_e2el_ms": m.std_e2el_ms,
        "percentiles_e2el_ms": _pairs(m.percentiles_e2el_ms),
    }
