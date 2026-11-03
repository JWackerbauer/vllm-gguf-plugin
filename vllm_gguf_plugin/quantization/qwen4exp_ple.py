# SPDX-License-Identifier: Apache-2.0
"""GGUF-quantized Qwen4Exp PLE (n-gram embedding) table.

vLLM's Qwen4Exp keeps the n-gram embedding table in pinned host memory and looks rows
up from the GPU through UVA (``--engram-config '{"cpu_offload": true}'``). Its own
formats are BF16 and FP8, i.e. 51-102 GB for Qwen3.8-Flash-Next; the GGUF stores the
table as IQ4_NL (27 GB). This method keeps the raw GGUF rows in the pinned table:

* the table is ``uint8 [rows, row_bytes]`` and the layer's ``embedding_dim`` is set
  to ``row_bytes``, so vLLM's raw-byte lookup kernel copies whole quantized rows;
* ``dequantize`` turns the looked-up rows into activations with the plugin's GGUF
  dequantize kernel (one row = one n-gram head's ``head_dim`` values);
* the table is filled straight from the GGUF file (chunked memmap copy) in
  ``process_weights_after_loading``; it never goes through the weight iterator,
  which would hold a second full copy in RAM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gguf
import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from .. import ops
from ..gguf_reader import get_gguf_reader

logger = init_logger("vllm.gguf_plugin.qwen4exp_ple")

_CHUNK_BYTES = 1 << 30


@dataclass(frozen=True)
class _GGUFTable:
    path: str
    offset: int  # absolute file offset of the tensor data
    weight_type: int
    rows: int
    values_per_row: int
    row_bytes: int


_SOURCE: _GGUFTable | None = None


def register_gguf_table(gguf_files, tensor_name: str) -> None:
    """Record where the PLE table lives; called by the adapter before model init."""
    global _SOURCE
    for path in gguf_files:
        reader = get_gguf_reader(path)
        for t in reader.tensors:
            if t.name == tensor_name:
                values, rows = (int(x) for x in t.shape)  # GGUF dims: innermost first
                block, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
                _SOURCE = _GGUFTable(
                    path=os.fspath(path), offset=int(t.data_offset), weight_type=int(t.tensor_type),
                    rows=rows, values_per_row=values, row_bytes=values // block * type_size,
                )
                logger.info("Qwen4Exp PLE table: %s, %d rows x %d values, %s (%.1f GiB)",
                            os.path.basename(path), rows, values, t.tensor_type.name,
                            rows * _SOURCE.row_bytes / 2**30)
                _patch_method_selection()
                return
    raise ValueError(f"{tensor_name} not found in {list(gguf_files)}")


def _patch_method_selection() -> None:
    from vllm.models.qwen4_exp.common import ngram_embedding as ne

    cls = ne.Qwen4ExpPLEEmbeddingMethod
    if getattr(cls, "_gguf_patched", False):
        return
    original = cls.from_quant_config

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        from .config import GGUFConfig

        if isinstance(quant_config, GGUFConfig):
            return _make_method(ne)()
        return original(quant_config, prefix, embedding_dtype)

    cls.from_quant_config = staticmethod(from_quant_config)
    cls._gguf_patched = True


def _make_method(ne):
    class GGUFPLEEmbeddingMethod(ne.Qwen4ExpPLEEmbeddingMethod):
        """Raw GGUF rows in the PLE table, dequantized after lookup."""

        def create_weights(self, layer: nn.Module, input_size_per_partition: int,
                           output_partition_sizes: list[int], input_size: int,
                           output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
            del input_size, output_size, params_dtype, extra_weight_attrs
            src = _SOURCE
            if src is None:
                raise RuntimeError("GGUF PLE table was not registered before model init")
            if input_size_per_partition != src.values_per_row:
                raise ValueError(f"PLE head_dim {input_size_per_partition} != GGUF row "
                                 f"width {src.values_per_row}")
            rows = sum(output_partition_sizes)
            if rows < src.rows:
                raise ValueError(f"PLE table has {rows} rows, GGUF {src.rows}")
            weight = nn.Parameter(
                layer.allocate_embedding_weight(rows, src.row_bytes, torch.uint8),
                requires_grad=False,
            )
            layer.register_parameter("weight", weight)
            # vLLM's pinned lookup copies embedding_dim * element_size bytes per row
            layer.embedding_dim = src.row_bytes
            layer.gguf_values_per_row = src.values_per_row
            layer.gguf_weight_type = src.weight_type

        def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
            return torch.index_select(layer.weight, 0, input_.reshape(-1)).view(
                *input_.shape, layer.weight.shape[1])

        def dequantize(self, layer: nn.Module, embeddings: torch.Tensor,
                       output_dtype: torch.dtype) -> torch.Tensor:
            row_bytes = layer.weight.shape[1]
            quant = embeddings.reshape(-1, row_bytes).contiguous()
            values = ops.ggml_dequantize(quant, layer.gguf_weight_type,
                                         layer.gguf_values_per_row, quant.shape[0], output_dtype)
            return values.reshape(*embeddings.shape[:-1], -1)

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            src = _SOURCE
            dst = layer.weight.data.view(-1)
            total = src.rows * src.row_bytes
            data = np.memmap(src.path, dtype=np.uint8, mode="r", offset=src.offset, shape=(total,))
            for start in range(0, total, _CHUNK_BYTES):
                end = min(start + _CHUNK_BYTES, total)
                dst[start:end].copy_(torch.from_numpy(np.ascontiguousarray(data[start:end])))
            dst[total:].zero_()
            del data
            logger.info("Filled Qwen4Exp PLE table from GGUF (%.1f GiB, %s)",
                        total / 2**30, "pinned host" if not dst.is_cuda else "device")

    return GGUFPLEEmbeddingMethod
