# SPDX-License-Identifier: Apache-2.0
"""Per-device capability checks for pipeline parallelism over mixed GPUs.

vLLM's ``current_platform.get_device_capability()`` and friends default to
``device_id=0``: in a worker for the second GPU that is still the first GPU. With
two different GPUs (RTX 5090 sm_120 + RTX 4070 Ti sm_89 as PP stages) the second
worker then picks kernels for the wrong architecture, e.g. Qwen4Exp's sm_90+ CuTe
DSL hyper-connection kernel ("no kernel image is available for execution").

Once CUDA is initialized, device 0 is taken to mean the worker's current device.
For a single GPU, and in processes that never touch CUDA, nothing changes.
"""

from __future__ import annotations

import functools

import torch


def patch_platform() -> None:
    from vllm.platforms import current_platform

    cls = type(current_platform)
    if getattr(cls, "_gguf_per_device_capability", False) or not current_platform.is_cuda():
        return

    def current_device(device_id: int) -> int:
        if device_id == 0 and torch.cuda.is_initialized():
            return torch.cuda.current_device()
        return device_id

    for name in ("get_device_capability", "get_device_name"):
        original = getattr(cls, name).__func__

        def wrapper(klass, device_id: int = 0, *, _original=original):
            return _original(klass, current_device(device_id))

        setattr(cls, name, classmethod(functools.wraps(original)(wrapper)))
    cls._gguf_per_device_capability = True

