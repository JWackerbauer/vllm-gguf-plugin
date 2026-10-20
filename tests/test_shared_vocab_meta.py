# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_gguf_plugin.loader import unquantized_vocab_weights_on_meta
from vllm_gguf_plugin.weights_adapter import Qwen35MtpGGUFAdapter


def create(module):
    UnquantizedEmbeddingMethod().create_weights(
        module, 64, [1000], 1000, 64, torch.bfloat16
    )
    return module.weight


def test_vocab_weights_on_meta_only_inside_context():
    with unquantized_vocab_weights_on_meta():
        inside = create(torch.nn.Module())
    outside = create(torch.nn.Module())
    assert inside.is_meta and inside.shape == (1000, 64)
    assert inside.input_dim == 1 and inside.output_dim == 0
    assert not outside.is_meta


def test_qwen35_mtp_draft_shares_vocab():
    assert Qwen35MtpGGUFAdapter.vocab_shared_with_target
