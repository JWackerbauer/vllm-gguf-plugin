# SPDX-License-Identifier: Apache-2.0

import gguf
import pytest
import torch
from gguf import GGMLQuantizationType as WeightType

import vllm_gguf_plugin.quantization.linear as gguf_linear


def run_dispatch(monkeypatch, quant_type, rows, num_tokens):
    """Run _fused_mul_mat_gguf with fake kernels; return which kernels ran."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    weight = torch.zeros(rows, 2 * type_size, dtype=torch.uint8)
    x = torch.zeros(num_tokens, 2 * block_size)
    calls = []

    def fake_matmul(name):
        def fn(w, x, qtype, nrows):
            calls.append(name)
            return torch.zeros(x.shape[0], nrows)

        return fn

    def fake_dequantize(w, qtype, m, n, dtype):
        calls.append("dequant")
        return torch.zeros(m, n, dtype=dtype)

    monkeypatch.setattr(gguf_linear.ops, "ggml_mul_mat_vec_a8", fake_matmul("mmvq"))
    monkeypatch.setattr(gguf_linear.ops, "ggml_mul_mat_a8", fake_matmul("mmq"))
    monkeypatch.setattr(gguf_linear.ops, "ggml_dequantize", fake_dequantize)

    out = gguf_linear._fused_mul_mat_gguf(x, weight, int(quant_type))
    assert out.shape == (num_tokens, rows)
    return calls


@pytest.mark.parametrize(
    "quant_type, rows",
    [
        # large i-quant weights used to leave MMVQ above 8 rows
        (WeightType.IQ3_S, 17408),
        (WeightType.IQ4_XS, 6144),
        (WeightType.Q4_K, 6144),
        (WeightType.Q8_0, 64),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 8, 16, 32])
def test_small_batches_use_mmvq(monkeypatch, quant_type, rows, num_tokens):
    # e.g. 4 sequences x (1 + 3 MTP draft tokens) = 16 rows per decode step
    assert run_dispatch(monkeypatch, quant_type, rows, num_tokens) == ["mmvq"]


def test_iquant_above_mmvq_limit_uses_dequant(monkeypatch):
    calls = run_dispatch(monkeypatch, WeightType.IQ3_S, 17408, 33)
    assert calls == ["dequant"]
