# SPDX-License-Identifier: Apache-2.0
"""Give LMCache a model name that identifies the weights and the KV format.

LMCache's multiprocess connector keys every cached chunk by (token hash,
model_config.model, rank). For a GGUF model, model_config.model is the HF
config source (e.g. --hf-config-path /hf-config), which several GGUF models
of one architecture share, and the KV cache dtype is not part of the key at
all. Two such models (or one model with fp8 and nvfp4 KV) then read each
other's cached KV for the same prompt prefix and generate from the wrong
attention state. The cache server and its disk tier outlive the vLLM
process, so this happens on the first request after switching models.

This adds the weights file (name and size) and the KV cache dtype to the name
LMCache sees. Only the connector's view changes; vLLM's model config does not.
"""

from __future__ import annotations

import copy
import os
from functools import wraps

from vllm.logger import init_logger

logger = init_logger("vllm.gguf_plugin.lmcache_identity")

CONNECTOR = "LMCacheMPConnector"
# vLLM's copy of the connector passes vllm_config to these adapter factories;
# LMCache's own copy (which vLLM uses when lmcache is installed) derives the
# name in get_dcp_decorated_model_name(vllm_config, kv_cache_config).
_NAME_HOOKS = (
    "get_dcp_decorated_model_name",
    "create_scheduler_adapter",
    "create_worker_adapter",
)


def lmcache_model_name(vllm_config) -> str:
    model_config = vllm_config.model_config
    parts = [str(model_config.model)]
    weights = getattr(model_config, "model_weights", "") or ""
    if weights and weights != model_config.model:
        try:
            size = f".{os.stat(weights).st_size:x}"
        except OSError:
            size = ""
        parts.append(os.path.basename(os.fspath(weights)) + size)
    parts.append(f"kv-{vllm_config.cache_config.cache_dtype}")
    # LMCache's disk tier uses "@" as the field separator in file names
    return "/".join(parts).replace("@", "_")


def _with_model_name(vllm_config, name: str):
    model_config = copy.copy(vllm_config.model_config)
    object.__setattr__(model_config, "model", name)
    patched = copy.copy(vllm_config)
    object.__setattr__(patched, "model_config", model_config)
    return patched


def _patch_module(module) -> None:
    if getattr(module, "_gguf_lmcache_identity_patched", False):
        return
    hooks = [n for n in _NAME_HOOKS if hasattr(module, n)]
    if not hooks:
        logger.warning(
            "%s: no known model-name hook; cached KV may be shared between "
            "GGUF models that use the same HF config",
            module.__name__,
        )
        return
    for fn_name in hooks:
        original = getattr(module, fn_name)
        # the vllm_config argument's position differs between the hooks
        pos = 0 if fn_name == "get_dcp_decorated_model_name" else 2

        @wraps(original)
        def hook(*args, _orig=original, _pos=pos, **kwargs):
            args = list(args)
            name = lmcache_model_name(args[_pos])
            logger.info("LMCache model name: %s", name)
            args[_pos] = _with_model_name(args[_pos], name)
            return _orig(*args, **kwargs)

        setattr(module, fn_name, hook)
    module._gguf_lmcache_identity_patched = True


def _patch_lmcache_connector() -> None:
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    loader = KVConnectorFactory._registry.get(CONNECTOR)
    if loader is None or getattr(loader, "_gguf_lmcache_identity", False):
        return

    def patched_loader():
        cls = loader()
        import sys

        _patch_module(sys.modules[cls.__module__])
        return cls

    patched_loader._gguf_lmcache_identity = True
    KVConnectorFactory._registry[CONNECTOR] = patched_loader
