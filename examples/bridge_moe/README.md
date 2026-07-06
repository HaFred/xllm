# REC xAttention for Qwen3-MoE (`examples/bridge_moe`)

Launch scripts for **REC xAttention + beam search** on **Qwen3-MoE-15B-A2B**
using the native `backend=rec` **LlmRec** path (not the OneRec encoder–decoder
path).

This matches the release note:

> REC XAttention for Qwen3-MoE on CUDA

## What runs

```text
REC (Python)
  -> RecMaster / LlmRecMasterPipeline
  -> FixedStepsScheduler / RecMultiRoundSchedulerPipeline
  -> RecEngine / RecMultiRoundEnginePipeline
  -> LlmRecMultiRoundPipeline
  -> Qwen3MoeForCausalLM (xLLM C++ weights, not HuggingFace forward)
  -> XAttentionImpl (shared + unshared KV, two-stage decode by default)
  -> rec_beam_search.cu + cache_select
```

**Required:** `max_decode_rounds > 0` (scripts default to `3`).

## Prerequisites

1. Build xLLM with CUDA and install the pybind extension:

   ```bash
   cd /path/to/xllm-frefork
   pip install -e .
   ```

2. Cache weights under `HF_HOME` (default `~/.cache/huggingface`):

   ```bash
   huggingface-cli download Qwen/Qwen3-MoE-15B-A2B
   ```

   Or point to an existing directory with `--model` / `XLLM_MODEL`.

3. GPU memory: 15B MoE + beam search is heavy. Start with `--smoke` or
   `XLLM_SMOKE=1` (beam 8). Increase `beam_width` once stable.

## Quick start

From the repo root:

```bash
# Resolve Qwen/Qwen3-MoE-15B-A2B from HF_HOME snapshots
bash examples/bridge_moe/launch_rec_beam_search.sh
```

Smoke test (small beam, good for first run):

```bash
XLLM_SMOKE=1 bash examples/bridge_moe/launch_rec_beam_search.sh
```

Explicit checkpoint (any local Qwen3-MoE HF folder):

```bash
XLLM_MODEL="$HF_HOME/hub/models--Qwen--Qwen3-MoE-15B-A2B/snapshots/<hash>" \
  bash examples/bridge_moe/launch_rec_beam_search.sh
```

Two GPUs with expert parallel:

```bash
XLLM_MODEL=/path/to/Qwen3-MoE-15B-A2B \
  bash examples/bridge_moe/launch_rec_beam_search_multi_gpu.sh
```

Direct Python:

```bash
python examples/bridge_moe/run_rec_beam_search_moe.py \
  --devices cuda:0 \
  --beam-width 32 \
  --max-decode-rounds 3 \
  --prompt "where is bejing?"
```

## Environment variables (`launch_rec_beam_search.sh`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `XLLM_MODEL` | (unset) | Local weights dir; else newest HF snapshot |
| `HF_HOME` | `~/.cache/huggingface` | Hub cache root |
| `XLLM_HF_REPO` | `Qwen/Qwen3-MoE-15B-A2B` | Repo id for snapshot lookup |
| `XLLM_REC_DEVICES` | `cuda:0` | Device list |
| `XLLM_EP_SIZE` | `1` | Expert parallel size (MoE) |
| `XLLM_BEAM_WIDTH` | `32` | Beam width |
| `XLLM_MAX_DECODE_ROUNDS` | `3` | Fixed decode rounds / xAttention enable |
| `XLLM_MAX_SEQS_PER_BATCH` | `1` | Batch concurrency |
| `XLLM_SMOKE` | `0` | Set `1` for beam 8 smoke config |

## Config files

- `config_qwen3_moe_15b_a2b.json` — default beam 32
- `config_qwen3_moe_15b_a2b_smoke.json` — beam 8 for memory checks

## vs `examples/bridge_bs`

| | `bridge_bs` | `bridge_moe` |
|---|---|---|
| Target | Qwen3 dense (0.6B demo) | **Qwen3-MoE-15B-A2B** |
| `model_type` | `qwen3` | `qwen3_moe` |
| HF_HOME resolver | no | yes |
| `ep_size` | default 1 | exposed for multi-GPU MoE |

Both use the same **LlmRecMultiRoundPipeline** + **xAttention** stack.

## vs OneRec xAttention

| | Qwen3-MoE (this dir) | OneRec |
|---|---|---|
| Pipeline | `LlmRecMultiRoundPipeline` | `OneRecXAttentionMasterPipeline` |
| Architecture | Decoder-only AR | Encoder–decoder |
| Input | prompt / token ids | `sparse_embedding` tensors |

## Related code

- `xllm/core/layers/cuda/xattention.cpp`
- `xllm/core/runtime/rec_worker_impl.cpp` (`LlmRecMultiRoundPipeline`)
- `xllm/models/llm/qwen3_moe.h` (full/unshared KV wiring)
- `xllm/pybind/rec.py` (`REC` Python API)
- `examples/bridge_bs/` — smaller Qwen3 dense example
