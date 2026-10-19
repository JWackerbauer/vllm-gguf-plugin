# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from gguf import GGMLQuantizationType as WeightType

import vllm_gguf_plugin.quantization.linear as gguf_linear
from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod


def make_layer(weight, weight_type, shard_weight_type=None):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight.shard_id = []
    layer.weight_type = SimpleNamespace(
        weight_type=int(weight_type), shard_weight_type=shard_weight_type or {}
    )
    return layer


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 4, 5, 12])
def test_small_unquantized_weight_is_transposed(dtype, num_tokens):
    # e.g. the merged in_proj_ba gate projection of a Qwen3.5 GatedDeltaNet layer
    torch.manual_seed(0)
    weight = torch.randn(96, 512, dtype=dtype)
    bias = torch.randn(96, dtype=dtype)
    x = torch.randn(num_tokens, 512, dtype=dtype)
    expected = x @ weight.T + bias
    wtype = WeightType.F32 if dtype == torch.float32 else WeightType.BF16
    layer = make_layer(weight.clone(), wtype, {0: int(wtype), 1: int(wtype)})

    method = GGUFLinearMethod(quant_config=None)
    method._transpose_small_unquantized_weight(layer)

    assert layer.weight_t.shape == (512, 96) and layer.weight_t.is_contiguous()
    assert torch.equal(layer.weight, weight)
    # same products, different summation order than x @ weight.T
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(
        method.apply(layer, x, bias), expected, rtol=tol, atol=tol
    )


def test_large_unquantized_weight_keeps_layout(monkeypatch):
    monkeypatch.setattr(gguf_linear, "_TRANSPOSE_MAX_NUMEL", 1000)
    layer = make_layer(torch.randn(64, 64), WeightType.F32)
    GGUFLinearMethod(quant_config=None)._transpose_small_unquantized_weight(layer)
    assert not hasattr(layer, "weight_t")


@pytest.mark.parametrize(
    "weight_type, shard_types",
    [
        (WeightType.Q8_0, {}),
        # an unquantized shard merged with a quantized one keeps the sliced path
        (WeightType.BF16, {0: int(WeightType.BF16), 1: int(WeightType.Q8_0)}),
    ],
)
def test_quantized_or_mixed_weights_keep_layout(weight_type, shard_types):
    layer = make_layer(
        torch.zeros(96, 512, dtype=torch.uint8), weight_type, shard_types
    )
    GGUFLinearMethod(quant_config=None)._transpose_small_unquantized_weight(layer)
    assert not hasattr(layer, "weight_t")
