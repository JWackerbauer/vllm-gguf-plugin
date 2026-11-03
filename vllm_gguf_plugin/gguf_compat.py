# SPDX-License-Identifier: Apache-2.0
"""Teach the gguf package quantization types newer than its release.

gguf 0.19 predates GGML_TYPE_Q2_0 (42), which ISTA-DASLab's GSQ-RCO GGUFs use (e.g.
the expert down-projections of Qwen3.8-Flash-Next Coder). GGUFReader rejects such
files outright ("42 is not a valid GGMLQuantizationType"). Register the type with its
block geometry so files load; the CUDA kernels handle Q2_0 themselves.
"""

from __future__ import annotations

import gguf

# name, value, (values per block, bytes per block)
_EXTRA_TYPES = (("Q2_0", 42, (64, 2 + 64 // 4)),)


def register_extra_quant_types() -> None:
    enum_cls = gguf.GGMLQuantizationType
    for name, value, sizes in _EXTRA_TYPES:
        member = enum_cls._value2member_map_.get(value)
        if member is None:
            member = int.__new__(enum_cls, value)
            member._name_ = name
            member._value_ = value
            member.__objclass__ = enum_cls
            enum_cls._member_map_[name] = member
            enum_cls._value2member_map_[value] = member
            enum_cls._member_names_.append(name)
            type.__setattr__(enum_cls, name, member)
        gguf.GGML_QUANT_SIZES.setdefault(member, sizes)


register_extra_quant_types()
