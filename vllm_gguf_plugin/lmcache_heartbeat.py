# SPDX-License-Identifier: Apache-2.0
"""Start LMCache's worker heartbeat when the KV caches are registered.

The LMCache multiprocess server reaps worker registrations that go silent: a
worker that has pinged gets ``--worker-reap-timeout-seconds`` (120 s), one that
never pinged gets ``--worker-registration-grace-seconds`` (3600 s). LMCache
0.5.6's worker adapter only starts its heartbeat on the first store, retrieve
or event poll. A server with no traffic in the first hour after startup (a
morning boot) therefore loses the live registration. When requests arrive, the
heartbeat starts in the healthy state and its pings succeed, so it never takes
the re-registration path: every lookup fails ("No GPU context found") and every
store is rejected ("unregistered GPU instance") until vLLM restarts.

Starting the heartbeat right after a successful registration makes the server
see pings from the start, so it only reaps a worker that has really stopped.
"""

from __future__ import annotations

from functools import wraps

from vllm.logger import init_logger

logger = init_logger("vllm.gguf_plugin.lmcache_heartbeat")


def patch_worker_adapter() -> None:
    try:
        from lmcache.integration.vllm import vllm_multi_process_adapter as adapter
    except ImportError:
        return
    cls = getattr(adapter, "LMCacheMPWorkerAdapter", None)
    if (
        cls is None
        or not hasattr(cls, "_ensure_heartbeat_started")
        or getattr(cls, "_gguf_eager_heartbeat", False)
    ):
        return
    original = cls.register_kv_caches

    @wraps(original)
    def register_kv_caches(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self._ensure_heartbeat_started()
        logger.info(
            "LMCache worker heartbeat started at registration (instance %s)",
            getattr(self, "instance_id", "?"),
        )
        return result

    cls.register_kv_caches = register_kv_caches
    cls._gguf_eager_heartbeat = True
