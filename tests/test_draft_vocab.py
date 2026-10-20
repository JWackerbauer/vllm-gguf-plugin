# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.models.interfaces import LocalArgmaxMixin

import vllm_gguf_plugin.quantization as gguf_quantization
from vllm_gguf_plugin.draft_vocab import load_draft_vocab, prune_draft_vocab

VOCAB, HIDDEN = 50, 16
SUBSET = [0, 3, 7, 11, 12, 20, 33, 41, 48, 49]


class FullHead(torch.nn.Module):
    """A single-rank unquantized lm_head as the GGUF plugin presents it."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_type = SimpleNamespace(weight_type=int(WeightType.F32))
        self.tp_size = 1
        self.shard_indices = SimpleNamespace(
            num_org_vocab_padding=0, org_vocab_start_index=0
        )
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: x @ layer.weight.T
        )


class DraftModel(LocalArgmaxMixin, torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = head
        # what vLLM's LogitsProcessor.get_top_tokens / _apply_head use
        self.logits_processor = SimpleNamespace(
            scale=1.0,
            soft_cap=None,
            head_dtype=None,
        )


@pytest.fixture
def draft(monkeypatch):
    # the GGUF matmul on CPU for an unquantized weight
    monkeypatch.setattr(
        gguf_quantization, "fused_mul_mat_gguf", lambda x, w, t: x @ w.T
    )
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    torch.manual_seed(0)
    full = FullHead(torch.randn(VOCAB, HIDDEN))
    model = DraftModel(full)
    lp = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(lp)
    lp.scale, lp.soft_cap, lp.head_dtype = 1.0, None, None
    model.logits_processor = lp
    return model, full


def test_top_tokens_are_the_argmax_over_the_subset(draft):
    model, full = draft
    hidden = torch.randn(6, HIDDEN)
    full_logits = hidden @ full.weight.T

    prune_draft_vocab(model, torch.tensor(SUBSET))

    expected = torch.tensor(SUBSET)[full_logits[:, SUBSET].argmax(dim=-1)]
    assert torch.equal(model.get_top_tokens(hidden), expected)
    assert model.lm_head.weight.shape == (len(SUBSET), HIDDEN)
    assert full.weight.shape == (VOCAB, HIDDEN)  # target's head untouched


def test_full_logits_are_masked_to_the_subset(draft):
    model, full = draft
    hidden = torch.randn(3, HIDDEN)
    full_logits = hidden @ full.weight.T

    prune_draft_vocab(model, torch.tensor(SUBSET))
    logits = model.compute_logits(hidden)

    assert logits.shape == (3, VOCAB)
    torch.testing.assert_close(logits[:, SUBSET], full_logits[:, SUBSET])
    outside = torch.ones(VOCAB, dtype=torch.bool)
    outside[SUBSET] = False
    assert torch.isinf(logits[:, outside]).all()


def test_load_draft_vocab_validates_ids(tmp_path):
    good = tmp_path / "ids.json"
    good.write_text(json.dumps([5, 1, 5, 3]))
    assert load_draft_vocab(str(good), VOCAB).tolist() == [1, 3, 5]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, VOCAB]))
    with pytest.raises(ValueError):
        load_draft_vocab(str(bad), VOCAB)
