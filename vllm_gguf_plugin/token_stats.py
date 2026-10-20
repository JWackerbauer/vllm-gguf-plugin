# SPDX-License-Identifier: Apache-2.0
"""Count generated token ids, to tune the reduced draft vocabulary to real use.

With VLLM_GGUF_TOKEN_STATS=/path/token_counts.npy the API server process adds
every generated token id to one counter per vocabulary entry and writes the
array atomically about once a minute (and at exit). Existing counts are loaded
first, so they accumulate across restarts. Only ids are counted: no text, no
prompts, nothing that ties a count to a request.

At startup the draft-vocabulary hook rebuilds the draft subset from these counts
(see draft_vocab.build_draft_subset).
"""

from __future__ import annotations

import atexit
import os
import threading
from functools import wraps

import numpy as np
from vllm.logger import init_logger

logger = init_logger("vllm.gguf_plugin.token_stats")

TOKEN_STATS_ENV = "VLLM_GGUF_TOKEN_STATS"
FLUSH_INTERVAL_S = 60.0
_INITIAL_SIZE = 1 << 18  # grows if a larger id shows up


def load_token_counts(path: str) -> np.ndarray | None:
    try:
        counts = np.load(path)
    except (OSError, ValueError):
        return None
    return counts.astype(np.int64, copy=False) if counts.ndim == 1 else None


class TokenCounter:
    def __init__(self, path: str):
        self.path = path
        loaded = load_token_counts(path)
        self.counts = (
            loaded if loaded is not None else np.zeros(_INITIAL_SIZE, np.int64)
        )
        self.lock = threading.Lock()
        self.dirty = False
        self._stop = threading.Event()
        threading.Thread(target=self._flush_loop, daemon=True).start()
        atexit.register(self.flush)

    def add(self, token_ids) -> None:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.size == 0:
            return
        with self.lock:
            top = int(ids.max())
            if top >= self.counts.size:
                grown = np.zeros(max(top + 1, 2 * self.counts.size), np.int64)
                grown[: self.counts.size] = self.counts
                self.counts = grown
            np.add.at(self.counts, ids, 1)
            self.dirty = True

    def flush(self) -> None:
        with self.lock:
            if not self.dirty:
                return
            snapshot = self.counts.copy()
            self.dirty = False
        tmp = f"{self.path}.tmp.npy"
        try:
            np.save(tmp, snapshot)
            os.replace(tmp, self.path)
        except OSError as e:
            logger.warning("Could not write token counts to %s: %s", self.path, e)

    def _flush_loop(self) -> None:
        while not self._stop.wait(FLUSH_INTERVAL_S):
            self.flush()


_counter: TokenCounter | None = None
_counter_lock = threading.Lock()


def _get_counter(path: str) -> TokenCounter:
    # Created on first use, so only the process that sees outputs (the API
    # server) ever reads or writes the file.
    global _counter
    if _counter is None:
        with _counter_lock:
            if _counter is None:
                _counter = TokenCounter(path)
                logger.info(
                    "Counting generated token ids into %s (%d already counted)",
                    path,
                    int(_counter.counts.sum()),
                )
    return _counter


def _patch_output_processor() -> None:
    path = os.environ.get(TOKEN_STATS_ENV)
    if not path:
        return
    from vllm.v1.engine import output_processor

    cls = output_processor.OutputProcessor
    if getattr(cls, "_gguf_token_stats_patched", False):
        return
    original = cls.process_outputs

    @wraps(original)
    def process_outputs(self, engine_core_outputs, *args, **kwargs):
        try:
            ids = [t for out in engine_core_outputs for t in (out.new_token_ids or ())]
            if ids:
                _get_counter(path).add(ids)
        except Exception as e:  # counting must never break serving
            logger.warning("Token counting failed: %s", e)
        return original(self, engine_core_outputs, *args, **kwargs)

    cls.process_outputs = process_outputs
    cls._gguf_token_stats_patched = True
