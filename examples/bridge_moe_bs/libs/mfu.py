"""Whole-run MFU estimation (vLLM v1/metrics/perf.py analytic formulas)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Per-byte SID vocab size for external linear heads.
EXTERNAL_HEAD_VOCAB = 8192

# H800 BF16/FP16 peak (bench.py default); override with --gpu-peak-tflops.
_DEFAULT_GPU_PEAK_TFLOPS = 990.0


@dataclass
class ExecutionContext:
    """Batch execution context for analytic FLOPs (vLLM perf.py compatible)."""

    num_prefill_requests: int = 0
    prefill_num_tokens: int = 0
    prefill_context_len: int = 0
    prefill_token_context_product: int = 0
    num_decode_requests: int = 0
    decode_num_tokens: int = 0
    decode_context_len: int = 0
    decode_token_context_product: int = 0

    def add(self, num_tokens: int, context_len: int, is_prefill: bool) -> None:
        if is_prefill:
            self.num_prefill_requests += 1
            self.prefill_num_tokens += num_tokens
            self.prefill_context_len += context_len
            self.prefill_token_context_product += num_tokens * context_len
        else:
            self.num_decode_requests += 1
            self.decode_num_tokens += num_tokens
            self.decode_context_len += context_len
            self.decode_token_context_product += num_tokens * context_len

    def total_num_tokens(self) -> int:
        return self.prefill_num_tokens + self.decode_num_tokens

    def total_token_context_product(self) -> int:
        return (
            self.prefill_token_context_product + self.decode_token_context_product
        )

    @classmethod
    def from_forward_batch(cls, batch_size: int, seq_len: int) -> ExecutionContext:
        """Full-sequence forward without KV cache: B independent prefill passes."""
        ctx = cls()
        for _ in range(batch_size):
            ctx.add(seq_len, seq_len, is_prefill=True)
        return ctx


@dataclass
class RunPerfTotals:
    """Accumulated FLOPs and memory traffic for the whole benchmark run."""

    total_flops: int = 0
    total_read_bytes: int = 0
    total_write_bytes: int = 0

    def observe(self, flops: int, read_bytes: int, write_bytes: int) -> None:
        self.total_flops += flops
        self.total_read_bytes += read_bytes
        self.total_write_bytes += write_bytes


@dataclass
class RunMfuMetrics:
    duration_s: float
    total_flops: int
    total_read_bytes: int
    total_write_bytes: int
    tflops_per_gpu: float
    gbps_per_gpu: float
    peak_gpu_tflops: float
    utilization: float


class ModelPerfEstimator:
    """Analytic per-forward FLOPs/memory model (vLLM v1/metrics/perf.py formulas)."""

    def __init__(
        self,
        cfg: dict,
        *,
        weight_bytes: int = 2,
        activation_bytes: int = 2,
        external_vocab: int = EXTERNAL_HEAD_VOCAB,
    ):
        self.L = int(cfg["num_hidden_layers"])
        self.D = int(cfg["hidden_size"])
        self.q = int(cfg["num_attention_heads"])
        self.kv = int(cfg["num_key_value_heads"])
        self.d = int(cfg["head_dim"])
        self.MI = int(cfg.get("moe_intermediate_size", cfg["intermediate_size"]))
        self.E = int(cfg.get("num_experts_per_tok", 0))
        self.Lm = self.L if int(cfg.get("num_experts", 0)) > 0 else 0
        self.Ld = self.L - self.Lm
        self.DI = int(cfg["intermediate_size"])
        self.S = int(cfg.get("num_shared_experts", 0) or 0)
        self.weight_bytes = weight_bytes
        self.activation_bytes = activation_bytes
        self.cache_bytes = activation_bytes
        self.external_vocab = external_vocab

    def estimate_backbone_forward(self, batch_size: int, seq_len: int) -> tuple[int, int, int]:
        ctx = ExecutionContext.from_forward_batch(batch_size, seq_len)
        flops = self._attention_flops(ctx) + self._ffn_flops(ctx)
        read_b, write_b = self._attention_bytes(ctx)
        ffn_read, ffn_write = self._ffn_bytes(ctx)
        return flops, read_b + ffn_read, write_b + ffn_write

    def estimate_external_head(self, batch_size: int) -> tuple[int, int, int]:
        V = self.external_vocab
        flops = 2 * batch_size * self.D * V
        read_b = batch_size * self.D * self.activation_bytes
        read_b += self.D * V * self.weight_bytes
        write_b = batch_size * V * self.activation_bytes
        return flops, read_b, write_b

    def _attention_flops(self, ctx: ExecutionContext) -> int:
        L, D, q, kv, d = self.L, self.D, self.q, self.kv, self.d
        T = ctx.total_num_tokens()
        TC = ctx.total_token_context_product()
        return (
            2 * T * D * (q + 2 * kv) * d * L
            + 2 * q * TC * d * L
            + 2 * q * TC * d * L
            + 2 * T * D * q * d * L
        )

    def _ffn_flops(self, ctx: ExecutionContext) -> int:
        Ld, Lm, D, DI, MI, E, S = self.Ld, self.Lm, self.D, self.DI, self.MI, self.E, self.S
        T = ctx.total_num_tokens()
        flops = 0
        if Ld:
            flops += 2 * D * 3 * DI * T * Ld
        if Lm and E:
            flops += 2 * D * 3 * MI * (T * E) * Lm
        if Lm and S:
            flops += 2 * D * 3 * MI * S * T * Lm
        return flops

    def _attention_bytes(self, ctx: ExecutionContext) -> tuple[int, int]:
        L, D, q, kv, d = self.L, self.D, self.q, self.kv, self.d
        T = ctx.total_num_tokens()
        read_b = 0
        read_b += T * D * self.activation_bytes * L
        read_b += int(D * (q + 2 * kv) * d * self.weight_bytes * L)
        if ctx.prefill_num_tokens > 0:
            read_b += (
                (ctx.prefill_num_tokens * q + 2 * ctx.prefill_context_len * kv)
                * d
                * self.activation_bytes
                * L
            )
        read_b += T * q * d * self.activation_bytes * L
        read_b += int(q * d * D * self.weight_bytes * L)
        write_b = (
            T * (q + 2 * kv) * d * self.activation_bytes * L
            + 2 * T * kv * d * self.cache_bytes * L
            + T * D * self.activation_bytes * L
        )
        return read_b, write_b

    def _ffn_bytes(self, ctx: ExecutionContext) -> tuple[int, int]:
        Ld, Lm, D, DI, MI, E, S = self.Ld, self.Lm, self.D, self.DI, self.MI, self.E, self.S
        T = ctx.total_num_tokens()
        num_activated_tokens = T * E if E else 0
        read_b = 0
        write_b = 0
        if Ld:
            read_b += int(T * D * self.activation_bytes * Ld)
            read_b += int(2 * D * DI * self.weight_bytes * Ld)
            read_b += int(2 * T * DI * self.activation_bytes * Ld)
            read_b += int(T * DI * self.activation_bytes * Ld)
            read_b += int(D * DI * self.weight_bytes * Ld)
            write_b += int(2 * T * DI * self.activation_bytes * Ld)
            write_b += int(T * DI * self.activation_bytes * Ld)
            write_b += int(T * D * self.activation_bytes * Ld)
        if Lm and E:
            read_b += int(num_activated_tokens * D * self.activation_bytes * Lm)
            read_b += int(2 * D * MI * self.weight_bytes * Lm)
            read_b += int(2 * num_activated_tokens * MI * self.activation_bytes * Lm)
            read_b += int(num_activated_tokens * MI * self.activation_bytes * Lm)
            read_b += int(D * MI * self.weight_bytes * Lm)
            write_b += int(2 * num_activated_tokens * MI * self.activation_bytes * Lm)
            write_b += int(num_activated_tokens * MI * self.activation_bytes * Lm)
            write_b += int(num_activated_tokens * D * self.activation_bytes * Lm)
        if Lm and S:
            read_b += int(T * D * self.activation_bytes * Lm)
            read_b += int(2 * D * MI * S * self.weight_bytes * Lm)
            read_b += int(2 * T * S * MI * self.activation_bytes * Lm)
            read_b += int(T * S * MI * self.activation_bytes * Lm)
            read_b += int(D * MI * S * self.weight_bytes * Lm)
            write_b += int(2 * T * S * MI * self.activation_bytes * Lm)
            write_b += int(T * S * MI * self.activation_bytes * Lm)
            write_b += int(T * S * D * self.activation_bytes * Lm)
        return read_b, write_b


def dtype_byte_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def default_peak_tflops(device: str) -> float:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        name = torch.cuda.get_device_name(device).lower()
        if "h100" in name or "h200" in name:
            return 1979.0
        if "h800" in name or "h20" in name:
            return 990.0
        if "a100" in name:
            return 312.0
    return _DEFAULT_GPU_PEAK_TFLOPS


def compute_run_mfu(
    totals: RunPerfTotals,
    duration_s: float,
    peak_gpu_tflops: float,
) -> RunMfuMetrics | None:
    if duration_s <= 0 or (
        totals.total_flops == 0
        and totals.total_read_bytes == 0
        and totals.total_write_bytes == 0
    ):
        return None
    tflops_per_gpu = totals.total_flops / duration_s / 1e12
    gbps_per_gpu = (
        (totals.total_read_bytes + totals.total_write_bytes) / duration_s / 1e9
    )
    utilization = tflops_per_gpu / peak_gpu_tflops if peak_gpu_tflops > 0 else 0.0
    return RunMfuMetrics(
        duration_s=duration_s,
        total_flops=totals.total_flops,
        total_read_bytes=totals.total_read_bytes,
        total_write_bytes=totals.total_write_bytes,
        tflops_per_gpu=tflops_per_gpu,
        gbps_per_gpu=gbps_per_gpu,
        peak_gpu_tflops=peak_gpu_tflops,
        utilization=utilization,
    )


def print_run_mfu(mfu: RunMfuMetrics) -> None:
    """Print whole-run MFU in vLLM perf.py log format plus utilization %."""
    print("{s:{c}^{n}}".format(s=" MFU (whole run) ", n=60, c="="))
    print(
        "MFU: {:.1f} TF/s/GPU {:.1f} GB/s/GPU".format(
            mfu.tflops_per_gpu, mfu.gbps_per_gpu
        )
    )
    print(
        "MFU utilization: {:.4f} ({:.2f}%)  [peak {:.0f} TFLOPS/GPU]".format(
            mfu.utilization, mfu.utilization * 100, mfu.peak_gpu_tflops
        )
    )
    print("=" * 60)


def mfu_to_dict(mfu: RunMfuMetrics) -> dict:
    return {
        "duration_s": mfu.duration_s,
        "total_flops": mfu.total_flops,
        "total_read_bytes": mfu.total_read_bytes,
        "total_write_bytes": mfu.total_write_bytes,
        "tflops_per_gpu": mfu.tflops_per_gpu,
        "gbps_per_gpu": mfu.gbps_per_gpu,
        "peak_gpu_tflops": mfu.peak_gpu_tflops,
        "utilization": mfu.utilization,
        "utilization_pct": mfu.utilization * 100,
    }


def record_backbone_forward(
    perf_totals: RunPerfTotals | None,
    perf_estimator: ModelPerfEstimator | None,
    seq: torch.Tensor,
) -> None:
    if perf_totals is None or perf_estimator is None:
        return
    batch_size, seq_len = int(seq.shape[0]), int(seq.shape[1])
    flops, read_b, write_b = perf_estimator.estimate_backbone_forward(batch_size, seq_len)
    perf_totals.observe(flops, read_b, write_b)


def record_external_head(
    perf_totals: RunPerfTotals | None,
    perf_estimator: ModelPerfEstimator | None,
    batch_size: int,
) -> None:
    if perf_totals is None or perf_estimator is None:
        return
    flops, read_b, write_b = perf_estimator.estimate_external_head(batch_size)
    perf_totals.observe(flops, read_b, write_b)


def record_recif_beam_search_batch(
    perf_totals: RunPerfTotals | None,
    perf_estimator: ModelPerfEstimator | None,
    *,
    batch_size: int,
    prefix_len: int,
    bf: tuple[int, int, int],
    beam: int,
) -> None:
    """Analytic FLOPs for one recif 3-step beam-search batch."""
    if perf_totals is None or perf_estimator is None:
        return
    bf0, bf1, _bf2 = bf
    keep = min(beam, bf0 * bf1)

    for active_batch, seq_len in (
        (batch_size, prefix_len),
        (batch_size * bf0, prefix_len + 1),
        (batch_size * keep, prefix_len + 2),
    ):
        flops, read_b, write_b = perf_estimator.estimate_backbone_forward(
            active_batch, seq_len
        )
        perf_totals.observe(flops, read_b, write_b)
        flops, read_b, write_b = perf_estimator.estimate_external_head(active_batch)
        perf_totals.observe(flops, read_b, write_b)
