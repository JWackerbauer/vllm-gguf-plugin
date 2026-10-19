# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from gguf import GGMLQuantizationType as WeightType

import vllm_gguf_plugin.quantization as gguf_quantization
from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod


def make_merged_layer(shard_types):
    """A merged layer as _create_padded_weight_param leaves it: shards stacked
    row-wise, each row padded to the widest shard's byte width."""
    shapes = {0: (4, 68), 1: (6, 36)}  # e.g. Q8_0 and Q4_0 rows of 64 values
    torch.manual_seed(0)
    padded = torch.zeros(10, 68, dtype=torch.uint8)
    offsets, start = {}, 0
    for idx, (rows, width) in shapes.items():
        padded[start : start + rows, :width] = torch.randint(0, 256, (rows, width))
        offsets[idx] = (start, start + rows, width)
        start += rows
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(padded, requires_grad=False)
    layer.weight.shard_id = [0, 1]
    layer.weight.shard_offset_map = offsets
    layer.weight_type = SimpleNamespace(
        weight_type=int(shard_types[0]),
        shard_weight_type={i: int(t) for i, t in enumerate(shard_types)},
    )
    return layer, padded.clone(), offsets


def run_apply(monkeypatch, layer, num_tokens=3):
    calls = []

    def fake_matmul(x, w, weight_type):
        calls.append((w, weight_type))
        return torch.zeros(x.shape[0], w.shape[0])

    monkeypatch.setattr(gguf_quantization, "fused_mul_mat_gguf", fake_matmul)
    out = GGUFLinearMethod(quant_config=None).apply(layer, torch.zeros(num_tokens, 64))
    return out, calls


def test_mixed_type_shards_are_packed_once(monkeypatch):
    layer, padded, offsets = make_merged_layer([WeightType.Q8_0, WeightType.Q4_0])
    GGUFLinearMethod(quant_config=None)._pack_mixed_type_shards(layer)

    assert layer.weight.numel() == 0  # padded copy freed
    packed = layer.weight_packed
    assert packed.numel() == 4 * 68 + 6 * 36

    out, calls = run_apply(monkeypatch, layer)
    assert out.shape == (3, 10)
    for (w, wtype), idx, expected_type in zip(
        calls, (0, 1), (WeightType.Q8_0, WeightType.Q4_0)
    ):
        start, end, width = offsets[idx]
        assert torch.equal(w, padded[start:end, :width])
        assert w.is_contiguous()
        # a view into the packed buffer, not a per-call copy
        assert packed.data_ptr() <= w.data_ptr() < packed.data_ptr() + packed.numel()
        assert wtype == int(expected_type)


def test_uniform_type_shards_are_left_alone(monkeypatch):
    layer, padded, _ = make_merged_layer([WeightType.Q8_0, WeightType.Q8_0])
    GGUFLinearMethod(quant_config=None)._pack_mixed_type_shards(layer)
    assert not hasattr(layer, "weight_packed")
    assert torch.equal(layer.weight, padded)
