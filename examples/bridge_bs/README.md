# REC beam search bridge (`examples/bridge_bs`)

This directory demonstrates **Option 2: full xLLM REC parity** — driving the native
`backend=rec` stack from Python instead of bridging individual CUDA ops into
HuggingFace.

## What runs under the hood

```text
REC (Python) -> RecMaster (pybind)
  -> FixedStepsScheduler
  -> RecEngine / LlmRecMultiRoundPipeline
  -> Qwen3MoeForCausalLM (xLLM C++ model, not HF)
  -> RecSampler top-k
  -> rec_beam_search.cu (CUDA beam_search + cache_select)
  -> xAttention shared/unshared KV
```

The decode-step branch in `rec_beam_search.cu` (combined_probs top-k +
`beam_search_step`) is exercised automatically when `max_decode_rounds > 0`.

## Prerequisites

- xLLM built with CUDA (`USE_CUDA`) and pybind extension `xllm_export`
- A supported LlmRec checkpoint on disk (`qwen2`, `qwen3`, `qwen3_moe`)
- GPU with enough memory for your `beam_width` (default 64)

Rebuild after pulling these changes:

```bash
pip install -e .
```

## Quick start

Text prompt:

```bash
python examples/bridge_bs/run_rec_beam_search.py \
  --model /path/to/Qwen3-0.6B \
  --devices cuda:0 \
  --prompt "where is bejing?"
```

Pre-tokenized prompt (same ids as `xllm/c_api/examples/test_query_rec_completions.cpp`):

```bash
python examples/bridge_bs/run_rec_beam_search.py \
  --model /path/to/Qwen3-0.6B \
  --token-prompt 2870,374,387,98168,30
```

## Python API

```python
from xllm import REC, BeamSearchParams

rec = REC(
    model="/path/to/Qwen3-0.6B",
    devices="cuda:0",
    beam_width=64,
    max_decode_rounds=3,
    block_size=1,
)

params = BeamSearchParams(beam_width=64, max_tokens=3, top_logprobs=64)
outputs = rec.beam_search("where is bejing?", params=params)

for seq in outputs[0].outputs:
    print(seq.index, seq.token_ids, seq.logprobs)

rec.finish()
```

`configure_rec_runtime()` is called internally before `RecMaster` construction
to set `FLAGS_*`, `RecConfig`, `BeamSearchConfig`, and `SchedulerConfig`
(mirrors `xllm/c_api/internal/rec.cpp`).

## vs LLM beam search (`examples/generate_beam_search.py`)

| | LLM (`backend=llm`) | REC (`backend=rec`) |
|---|---|---|
| Scheduler | Continuous batching | Fixed decode rounds |
| Beam kernel | NPU `beam_searcher.cpp` / CPU fallback on CUDA | `rec_beam_search.cu` |
| KV layout | Standard paged KV | xAttention shared + unshared |
| Typical models | General causal LM | Qwen2/3 LlmRec family |

## Optional config file

`config_example.json`:

```json
{
  "beam_width": 64,
  "max_tokens": 3,
  "top_logprobs": 64,
  "temperature": 1.0,
  "top_k": 64
}
```

```bash
python examples/bridge_bs/run_rec_beam_search.py \
  --model /path/to/Qwen3-0.6B \
  --config examples/bridge_bs/config_example.json
```

## Related code

- CUDA kernel: `xllm/core/kernels/cuda/rec_beam_search.cu`
- Worker loop: `xllm/core/runtime/rec_worker_impl.cpp` (`execute_beam_search`)
- Pybind: `xllm/pybind/bind.cpp` (`RecMaster`, `configure_rec_runtime`)
- Python wrapper: `xllm/pybind/rec.py`
- Design doc: `docs/en/design/generative_recommendation_design.md`
