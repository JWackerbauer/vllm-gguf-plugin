# SPDX-License-Identifier: Apache-2.0
"""One parsed GGUFReader per file and process.

Constructing a gguf.GGUFReader parses every metadata field, including the
tokenizer's token and merge arrays: ~8 s for a GGUF with a 248k-token vocabulary.
Loading a model asked for a fresh reader of the same file 9-10 times (tensor
names, config patching, the weights adapter, the loader itself, and all of it
again for an MTP draft model), which made up ~78 s of a 110 s model load.
Readers are read-only (memory-mapped), so one instance can be shared.
"""

import functools
import os

import gguf


@functools.lru_cache(maxsize=8)
def _cached_reader(path: str, mtime_ns: int, size: int) -> gguf.GGUFReader:
    del mtime_ns, size  # part of the cache key: a changed file is parsed again
    return gguf.GGUFReader(path)


def get_gguf_reader(path: str | os.PathLike) -> gguf.GGUFReader:
    """A gguf.GGUFReader for path, parsed once per process."""
    real = os.path.realpath(os.fspath(path))
    st = os.stat(real)
    return _cached_reader(real, st.st_mtime_ns, st.st_size)
