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

"""Python entry for xLLM generative recommendation (backend=rec)."""

from __future__ import annotations

import os
import threading
import uuid
from typing import Callable, List, Optional, Sequence, Union

import xllm_export
from xllm_export import Options, RecMaster, RequestOutput, RequestParams

from . import utils
from .errors import ValidationError
from .params import BeamSearchParams, to_request_params

_REQUEST_PARAM_FIELDS = (
    "service_request_id",
    "x_request_id",
    "x_request_time",
    "max_tokens",
    "n",
    "best_of",
    "echo",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "temperature",
    "top_p",
    "top_k",
    "logprobs",
    "top_logprobs",
    "skip_special_tokens",
    "ignore_eos",
    "is_embeddings",
    "stop",
    "stop_token_ids",
    "beam_width",
    "num_return_sequences",
    "add_special_tokens",
    "is_sample_request",
    "sample_slots",
)


def _clone_request_params(src: RequestParams) -> RequestParams:
    dst = RequestParams()
    for field in _REQUEST_PARAM_FIELDS:
        setattr(dst, field, getattr(src, field))
    return dst


class REC:
    """Offline REC client using the native xLLM RecMaster stack.

    This drives the full ``backend=rec`` path (FixedStepsScheduler,
    LlmRecMultiRoundPipeline, ``rec_beam_search.cu``, xAttention KV, etc.).
    It does not use HuggingFace Transformers for forward.
    """

    def __init__(
        self,
        model: str,
        devices: str = "cuda:0",
        *,
        beam_width: int = 64,
        max_decode_rounds: int = 3,
        block_size: int = 1,
        max_cache_size: int = 0,
        max_memory_utilization: float = 0.8,
        max_tokens_per_batch: int = 8192,
        max_seqs_per_batch: int = 4,
        enable_prefix_cache: bool = False,
        enable_chunked_prefill: bool = False,
        enable_graph: bool = False,
        enable_rec_fast_sampler: bool = True,
        rec_worker_max_concurrency: int = 1,
        master_node_addr: str = "",
        nnodes: int = 1,
        node_rank: int = 0,
        dp_size: int = 1,
        ep_size: int = 1,
        disable_log_stats: bool = True,
        **kwargs: object,
    ) -> None:
        if kwargs:
            unknown = ", ".join(sorted(str(k) for k in kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        if not os.path.exists(model):
            raise ValueError(f"model path does not exist: {model}")

        model_type = utils._infer_model_type(model)
        utils._configure_cpp_chat_template(True, model_type)

        configure_rec = getattr(xllm_export, "configure_rec_runtime", None)
        if not callable(configure_rec):
            raise RuntimeError(
                "xllm_export.configure_rec_runtime is missing; rebuild xllm_export."
            )
        configure_rec(
            max_decode_rounds,
            beam_width,
            max_seqs_per_batch,
            max_tokens_per_batch,
            block_size,
            enable_rec_fast_sampler,
            enable_chunked_prefill,
        )

        options = Options()
        options.model_path = model
        options.task_type = "generate"
        options.devices = devices
        options.backend = "rec"
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        options.enable_prefix_cache = enable_prefix_cache
        options.max_tokens_per_batch = max_tokens_per_batch
        options.max_seqs_per_batch = max_seqs_per_batch
        options.enable_chunked_prefill = enable_chunked_prefill
        options.beam_width = beam_width
        options.rec_worker_max_concurrency = rec_worker_max_concurrency
        options.enable_graph = enable_graph
        options.enable_offline_inference = True
        options.disable_log_stats = disable_log_stats
        options.spawn_worker_path = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )
        options.nnodes = nnodes
        options.node_rank = node_rank
        options.dp_size = dp_size
        options.ep_size = ep_size
        if master_node_addr:
            options.master_node_addr = master_node_addr
        else:
            options.master_node_addr = f"127.0.0.1:{utils.get_free_port()}"

        self._beam_width = beam_width
        self._max_decode_rounds = max_decode_rounds
        self.master = RecMaster(options)

    def start_profile(self) -> bool:
        return bool(self.master.start_profile())

    def stop_profile(self) -> bool:
        return bool(self.master.stop_profile())

    def finish(self) -> None:
        try:
            utils.terminate_process(os.getpid())
        except Exception:
            pass

    def _prepare_request_params(
        self,
        params: Optional[Union[RequestParams, BeamSearchParams]],
    ) -> RequestParams:
        request_params = to_request_params(params, default_cls=BeamSearchParams)
        if request_params.beam_width <= 0:
            request_params.beam_width = self._beam_width
        request_params.logprobs = True
        if request_params.top_logprobs <= 0:
            request_params.top_logprobs = request_params.beam_width
        if request_params.max_tokens <= 0:
            request_params.max_tokens = self._max_decode_rounds
        return request_params

    def beam_search_tokens(
        self,
        token_prompts: Sequence[Sequence[int]],
        params: Optional[Union[RequestParams, BeamSearchParams]] = None,
    ) -> List[RequestOutput]:
        """Run REC multi-round beam search on pre-tokenized prompts."""
        request_params = self._prepare_request_params(params)
        outputs: List[Optional[RequestOutput]] = [None] * len(token_prompts)
        lock = threading.Lock()

        def make_callback(index: int) -> Callable[[RequestOutput], bool]:
            def callback(output: RequestOutput) -> bool:
                with lock:
                    outputs[index] = output
                return True

            return callback

        for index, tokens in enumerate(token_prompts):
            per_request = _clone_request_params(request_params)
            per_request.request_id = str(uuid.uuid4())
            self.master.handle_token_request(
                list(tokens),
                per_request,
                make_callback(index),
            )

        self.master.generate()

        results: List[RequestOutput] = []
        for index, output in enumerate(outputs):
            if output is None:
                raise RuntimeError(f"REC request {index} produced no output")
            if output.status is not None and not output.status.ok:
                raise ValidationError(output.status.code, output.status.message)
            results.append(output)
        return results

    def beam_search(
        self,
        prompts: Union[str, Sequence[str]],
        params: Optional[Union[RequestParams, BeamSearchParams]] = None,
    ) -> List[RequestOutput]:
        """Run REC beam search on text prompts (LlmRec / Qwen3 family)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        request_params = self._prepare_request_params(params)
        outputs: List[Optional[RequestOutput]] = [None] * len(prompts)
        lock = threading.Lock()

        def make_callback(index: int) -> Callable[[RequestOutput], bool]:
            def callback(output: RequestOutput) -> bool:
                with lock:
                    outputs[index] = output
                return True

            return callback

        for index, prompt in enumerate(prompts):
            per_request = _clone_request_params(request_params)
            per_request.request_id = str(uuid.uuid4())
            self.master.handle_prompt_request(
                prompt,
                per_request,
                make_callback(index),
            )

        self.master.generate()

        results: List[RequestOutput] = []
        for index, output in enumerate(outputs):
            if output is None:
                raise RuntimeError(f"REC request {index} produced no output")
            if output.status is not None and not output.status.ok:
                raise ValidationError(output.status.code, output.status.message)
            output.prompt = prompts[index]
            results.append(output)
        return results
