# SPDX-License-Identifier: Apache-2.0
"""Pipeline parallelism for Qwen4Exp when its PLE layers sit on the first rank.

vLLM rejects PP for Qwen4Exp models with an n-gram PLE embedding because non-first
pipeline ranks do not receive the raw ``input_ids`` PLE needs. The model itself only
runs PLE through ``layer.ple`` on the layers a rank owns, and the n-gram context is
built from the per-request token state every rank has. So PP works as long as every
PLE layer is on rank 0, as for Qwen3.8-Flash-Next (one PLE layer, the second).

Both guards (the config check and the runner's model state) are relaxed only then;
any other layout still fails as before.
"""

from __future__ import annotations

import copy
import functools

from vllm.logger import init_logger

logger = init_logger("vllm.gguf_plugin.qwen4exp_pp")


def ple_layers_on_first_rank(vllm_config) -> bool:
    """True if PP > 1 and every PLE layer is within the first pipeline stage."""
    from vllm.distributed.utils import get_pp_indices

    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    text_config = vllm_config.model_config.hf_text_config
    ple_ids = getattr(text_config, "ple_layer_ids", None) or ()
    if pp_size <= 1 or not ple_ids:
        return False
    start, end = get_pp_indices(text_config.num_hidden_layers, 0, pp_size)
    # ple_layer_ids are 1-based: layer i has PLE if (i + 1) in ple_layer_ids
    return all(start <= abs_id - 1 < end for abs_id in ple_ids)


def _with_single_stage(vllm_config):
    """Shallow copy of ``vllm_config`` whose parallel config reports PP=1."""
    parallel = copy.copy(vllm_config.parallel_config)
    object.__setattr__(parallel, "pipeline_parallel_size", 1)
    patched = copy.copy(vllm_config)
    object.__setattr__(patched, "parallel_config", parallel)
    return patched


def patch_config_check() -> None:
    """Relax the PLE/PP check in Qwen4Exp's ``verify_and_update_config``."""
    try:
        from vllm.model_executor.models import config as model_configs
    except ImportError:
        return
    cls = getattr(model_configs, "Qwen4ExpForConditionalGenerationConfig", None)
    if cls is None or getattr(cls, "_gguf_pp_patched", False):
        return
    original = cls.verify_and_update_config

    @functools.wraps(original)
    def verify_and_update_config(vllm_config) -> None:
        if not ple_layers_on_first_rank(vllm_config):
            return original(vllm_config)
        text_config = vllm_config.model_config.hf_text_config
        ple_ids = text_config.ple_layer_ids
        # the check keys on ple_layer_ids; hide them for the duration of the call
        # (the only other use, the DBO/microbatching check, also keys on QSA)
        text_config.ple_layer_ids = []
        try:
            original(vllm_config)
        finally:
            text_config.ple_layer_ids = ple_ids
        logger.info(
            "Qwen4Exp: PLE layers %s are on the first pipeline stage, "
            "allowing pipeline_parallel_size=%d",
            ple_ids,
            vllm_config.parallel_config.pipeline_parallel_size,
        )

    cls.verify_and_update_config = staticmethod(verify_and_update_config)
    cls._gguf_pp_patched = True


def patch_model_state() -> None:
    """Relax the same check in the GPU runner's Qwen4Exp model state."""
    from vllm.models.qwen4_exp.nvidia import model_state

    cls = model_state.Qwen4ExpModelState
    if getattr(cls, "_gguf_pp_patched", False):
        return
    original = cls.__init__

    @functools.wraps(original)
    def __init__(self, vllm_config, *args, **kwargs):
        if ple_layers_on_first_rank(vllm_config):
            vllm_config = _with_single_stage(vllm_config)
        original(self, vllm_config, *args, **kwargs)

    cls.__init__ = __init__
    cls._gguf_pp_patched = True


def patch_kv_cache_tensors() -> None:
    """Drop KV cache tensors for layers on other pipeline stages.

    The KV cache groups are unified over all stages, so a cache type that only one
    stage has (Qwen4Exp's PLE conv state, on the first stage) is an empty group on
    the others, while their tensor list still names the other stage's layer, and
    ``allocate_kv_cache`` fails looking for it (StopIteration). Per worker, keep
    only the layers this worker's groups contain.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:
        return
    if getattr(Worker, "_gguf_pp_kv_patched", False):
        return
    original = Worker.initialize_from_config

    @functools.wraps(original)
    def initialize_from_config(self, kv_cache_config, *args, **kwargs):
        own = {n for g in kv_cache_config.kv_cache_groups for n in g.layer_names}
        kept = []
        for tensor in kv_cache_config.kv_cache_tensors:
            layers = [n for n in tensor.layers if n in own]
            if layers:
                if len(layers) != len(tensor.layers):
                    tensor = copy.copy(tensor)
                    object.__setattr__(tensor, "layers", layers)
                kept.append(tensor)
        if len(kept) != len(kv_cache_config.kv_cache_tensors):
            logger.info(
                "Dropped %d KV cache tensor(s) of other pipeline stages",
                len(kv_cache_config.kv_cache_tensors) - len(kept),
            )
            kv_cache_config = copy.copy(kv_cache_config)
            object.__setattr__(kv_cache_config, "kv_cache_tensors", kept)
        return original(self, kv_cache_config, *args, **kwargs)

    Worker.initialize_from_config = initialize_from_config
    Worker._gguf_pp_kv_patched = True
