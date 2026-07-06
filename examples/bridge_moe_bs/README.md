# recif xLLM bridge (`examples/bridge_moe_bs`)

Bridge the **recif** Megatron SID checkpoint workflow to xLLM's native REC beam
search stack on **Qwen3-MoE**, without modifying code outside the xLLM repo.

This directory is the recif-specific counterpart to `examples/bridge_bs` (dense
Qwen3 demo). Both use the same **LlmRecMultiRoundPipeline** + **xAttention**
CUDA path; here the model is a exported recif Qwen3-MoE checkpoint with 3
external SID heads.

## What runs (`--backend xllm`)

```text
recif_bs.py / fred_run.sh
  -> REC (Python, xllm/pybind/rec.py)
  -> RecMaster / LlmRecMasterPipeline
  -> FixedStepsScheduler / RecMultiRoundSchedulerPipeline
  -> RecEngine / RecMultiRoundEnginePipeline
  -> LlmRecMultiRoundPipeline
  -> Qwen3MoeForCausalLM + recif external SID heads (C++ weights, not HF forward)
  -> XAttentionImpl (shared + unshared KV, two-stage decode by default)
  -> rec_beam_search.cu + cache_select
```

**Required for xAttention multi-round decode:** `max_decode_rounds > 0` (default
`3` for recif's 3 byte-level SID steps).

With `--backend hf`, the same exported weights are driven through transformers +
3 external `Linear` heads in Python (`recif_inference.py`) for correctness
reference and MFU baselines.

## Files

| File | Purpose |
|---|---|
| `recif_common.py` | Megatron->HF mapping, SID token layout helpers |
| `export_recif_to_hf.py` | One-time checkpoint export utility |
| `recif_inference.py` | Correct 3-head SID beam search (HF/transformers backend) |
| `recif_bs.py` | CLI runner (`--backend hf` or `--backend xllm`) |
| `fred_run.sh` | Container helper matching the HF benchmark flags |
| `relink_xllm_export.sh` | Incremental C++ rebuild without CMake reconfigure |
| `libs/mfu.py`, `libs/perf_slo.py` | Whole-run MFU + bench.py-compatible SLO metrics |

## Prerequisites

1. xLLM built with CUDA (`USE_CUDA`) and pybind extension `xllm_export`:

   ```bash
   cd /path/to/xllm
   pip install -e .
   ```

2. Megatron recif checkpoint mounted in the container (e.g. at `/recif/checkpoint`).

3. GPU memory: MoE + beam 1000 is heavy. Use smaller `--beam` / `RECIF_BEAM` for
   smoke tests before full benchmarks.

4. Inside the `xllm-cuda` container, set `PYTHONPATH` to the build tree when
   using `--backend xllm` (`fred_run.sh` does this automatically).

## Quick start (inside `xllm-cuda`)

```bash
cd /workspace/xllm/examples/bridge_moe_bs

# Mount recif at /recif, then:
./fred_run.sh
```

Or step by step:

```bash
# 1) Export Megatron checkpoint (once)
python export_recif_to_hf.py \
  --ckpt /recif/checkpoint \
  --config /recif/checkpoint/config.json \
  --out-dir /workspace/xllm/examples/bridge_moe_bs/xllm_hf_export \
  --fused-experts

# 2) Run recif beam search
python recif_bs.py \
  --backend xllm \
  --model /workspace/xllm/examples/bridge_moe_bs/xllm_hf_export \
  --history 598080194427,628177754964,755993681678 \
  --bf 64,128,256 \
  --beam 1000 \
  --device cuda:1 \
  --warmup 10 \
  --num-requests 64 \
  --batch-size 4 \
  --out-dir out-xllm/64requests-xllm \
  --just-log-one-output
```

### `fred_run.sh` environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `RECIF_BACKEND` | `xllm` | `hf` or `xllm` |
| `RECIF_CKPT` | `/recif/checkpoint` | Megatron checkpoint dir (export only) |
| `RECIF_MODEL` | `./xllm_hf_export` | Exported HF weights dir (must include `model.safetensors`; export also copies tokenizer files) |
| `RECIF_HISTORY` | demo item IDs | Comma-separated history |
| `RECIF_BF` | `64,128,256` | Branching factors per SID level |
| `RECIF_BEAM` | `1000` | Final beam width |
| `RECIF_DEVICE` | `cuda:1` | CUDA device |
| `RECIF_WARMUP` | `10` | Warmup requests |
| `RECIF_NUM_REQUESTS` | `64` | Benchmark request count |
| `RECIF_BATCH_SIZE` | `4` | xLLM batch size |
| `RECIF_MAX_DECODE_ROUNDS` | `3` | REC multi-round decode steps |
| `RECIF_OUT_DIR` | `out-xllm/64requests-${RECIF_BACKEND}` | Output directory |

## Export layout

```
xllm_hf_export/
  config.json                      # vocab_size=24576 + recif metadata
  model.safetensors                # Qwen3-MoE backbone (Megatron rank-0)
  recif_external_heads.safetensors   # heads.0/1/2 weights (8192 each)
  recif_export_meta.json
```

## Rebuild after C++ changes (important)

**Do not run `ninja xllm_export` in the existing build tree.** Ninja will try to
regenerate `build.ninja` via CMake, which fails in this container with:

```text
Could not find a package configuration file provided by "yalantinglibs"
```

The original full build installed Mooncake/yalantinglibs deps; a CMake regen
does not pick them up unless you re-run `sh third_party/dependencies.sh` and
export `CMAKE_PREFIX_PATH=/usr/local/yalantinglibs`.

For small C++ patches (recif external heads, `kv_seq_lens`, etc.), use the
incremental relink helper instead:

```bash
cd /workspace/xllm/examples/bridge_moe_bs
./relink_xllm_export.sh \
  xllm/core/framework/config/rec_config.cpp \
  xllm/core/framework/hf_model_loader.cpp \
  xllm/core/runtime/rec_worker_impl.cpp
```

This recompiles the listed sources from `compile_commands.json`, updates the
relevant `.a` archives, and relinks `xllm_export.so` without touching CMake.

## Backends and limitations

- **`--backend hf` (default in `recif_bs.py`)** — correct recif beam search using
  exported backbone + `recif_external_heads.safetensors`. Matches the HF
  `recif_beam_search_*` reference. Per-step TTFT/ITL are measured directly.
- **`--backend xllm`** — xLLM REC (`LlmRecMultiRoundPipeline`) with recif external
  SID heads when `config.json` contains a `recif` block. Rebuild `xllm_export.so`
  after C++ changes (see relink section above). Per-round TTFT/ITL are estimated
  from batch E2EL using HF-measured round ratios (xLLM does not expose per-round
  timestamps yet).
- **Per-round `--bf`** — fully honored on `--backend hf`. On `--backend xllm`,
  branching is approximated via `top_k=max(bf)`.
- **MFU** — analytic whole-run estimate (same formulas as HF reference) for both
  backends; written to `perf.json`.

## HF vs xLLM command mapping

| HF `recif_beam_search_concurrent.py` | xLLM `recif_bs.py` |
|---|---|
| `--ckpt ./checkpoint` | `--ckpt` (auto-exports) or `--model ./xllm_hf_export` |
| `--config ./checkpoint/config.json` | same / embedded in export dir |
| `--device cuda:1` | `--device cuda:1` |
| `--bf 64,128,256` | same (approximated on xllm) |
| `--beam 1000` | same |
| `--num-requests 64` | same |
| `--batch-size 4` | same |
| HF `model()` forward | xLLM `REC.beam_search_tokens()` |

## Recif C++ support (xLLM REC)

When `config.json` includes `"recif": { ... }`:

- `RecConfig::load_recif_from_model_config()` parses metadata at model load time
- `Qwen3MoeForCausalLM` registers 3 `Linear(hidden, 8192)` heads and loads
  `recif_external_heads.safetensors`
- `LlmRecMultiRoundPipeline` calls `logits_for_decode_round(..., round)` and applies
  `+8192` / `+16384` token offsets on decode rounds 2 and 3

## vs other examples

| | `bridge_bs` | `bridge_moe_bs` (this dir) |
|---|---|---|
| Target | Qwen3 dense (0.6B demo) | **recif Qwen3-MoE** SID model |
| `model_type` | `qwen3` | `qwen3_moe` + recif heads |
| Task | generic text/tokens REC demo | hierarchical SID beam search |
| Weights | stock HF Qwen3 | Megatron export + external heads |

| | Qwen3-MoE LlmRec (this dir) | OneRec |
|---|---|---|
| Pipeline | `LlmRecMultiRoundPipeline` | `OneRecXAttentionMasterPipeline` |
| Architecture | Decoder-only AR | Encoder–decoder |
| Input | token ids / prompt | `sparse_embedding` tensors |

## Related code

- `xllm/core/layers/cuda/xattention.cpp`
- `xllm/core/runtime/rec_worker_impl.cpp` (`LlmRecMultiRoundPipeline`)
- `xllm/models/llm/qwen3_moe.h` (MoE + recif external heads)
- `xllm/pybind/rec.py` (`REC` Python API)
- `examples/bridge_bs/` — smaller Qwen3 dense REC example
