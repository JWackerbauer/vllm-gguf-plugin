# SPDX-License-Identifier: Apache-2.0
"""Speculative drafting over a reduced vocabulary (FR-Spec style).

An MTP draft head computes logits over the whole vocabulary at every draft step,
which reads the full lm_head each time: with 4 draft tokens the output matrix is
read 5 times per decode step (once in verification). For a 27B GGUF with a
248k-token vocabulary that is ~16% of all weight bytes read per step.

Set VLLM_GGUF_DRAFT_VOCAB to a JSON list of token ids (e.g. the 32k most frequent
tokens) and enable "use_local_argmax_reduction" in --speculative-config. After the
draft model is loaded, its lm_head is replaced by one holding only those rows
(still GGUF-quantized), and draft_id_to_target_id maps the draft argmax back to
target token ids (vLLM's LocalArgmaxMixin does the mapping).

Verification still runs on the target model's full vocabulary, so outputs are
unchanged; only draft acceptance can drop, when the target's token is outside
the subset. Drafting must be greedy (vLLM's default draft_sample_method).
"""

from __future__ import annotations

import json
import os
from functools import wraps

import numpy as np
import torch
from vllm.logger import init_logger

# under the vllm namespace: vLLM only attaches log handlers to `vllm.*` loggers
logger = init_logger("vllm.gguf_plugin.draft_vocab")

DRAFT_VOCAB_ENV = "VLLM_GGUF_DRAFT_VOCAB"


class _ShardIndices:
    # what LogitsProcessor.get_top_tokens reads from a (single-rank) lm_head
    num_org_vocab_padding = 0
    org_vocab_start_index = 0


class _PrunedHeadMethod:
    def apply(self, layer, x, bias=None):
        from .quantization import fused_mul_mat_gguf

        out = fused_mul_mat_gguf(x, layer.weight, layer.gguf_weight_type)
        if bias is not None:
            out = out + bias[layer.token_ids]
        return out


class PrunedLMHead(torch.nn.Module):
    """The rows of a GGUF lm_head for a subset of token ids."""

    def __init__(self, lm_head: torch.nn.Module, token_ids: torch.Tensor):
        super().__init__()
        weight = lm_head.weight
        token_ids = token_ids.to(weight.device)
        self.register_buffer(
            "weight", weight.data.index_select(0, token_ids).contiguous(), False
        )
        self.register_buffer("token_ids", token_ids, False)
        self.gguf_weight_type = int(lm_head.weight_type.weight_type)
        self.tp_size = 1
        self.shard_indices = _ShardIndices()
        self.quant_method = _PrunedHeadMethod()


def load_draft_vocab(path: str, vocab_size: int) -> torch.Tensor:
    with open(path) as f:
        ids = sorted({int(i) for i in json.load(f)})
    if not ids or ids[0] < 0 or ids[-1] >= vocab_size:
        raise ValueError(f"{path}: token ids must be in [0, {vocab_size})")
    return torch.tensor(ids, dtype=torch.long)


# Rebuild the subset from real counts only once this many tokens were counted.
MIN_COUNTED_TOKENS = 100_000


def load_ranked_ids(path: str) -> list[int]:
    """Token ids in file order (most important first), without duplicates."""
    with open(path) as f:
        return list(dict.fromkeys(int(i) for i in json.load(f)))


def build_draft_subset(
    base_ranked: list[int], counts: np.ndarray, k: int, min_count: int = 2
) -> list[int]:
    """The k tokens to draft over: real-use tokens by count (seen at least
    min_count times), then the base ranking for the rest. Sorted ids."""
    seen = np.flatnonzero(counts >= min_count)
    by_count = seen[np.argsort(-counts[seen], kind="stable")]
    chosen = dict.fromkeys(int(t) for t in by_count[:k])
    for t in base_ranked:
        if len(chosen) >= k:
            break
        chosen.setdefault(t)
    return sorted(chosen)


def _coverage(ids, counts: np.ndarray) -> float:
    ids = np.asarray([i for i in ids if i < counts.size], dtype=np.int64)
    total = counts.sum()
    return float(counts[ids].sum() / total) if total else float("nan")


def select_draft_vocab(path: str, vocab_size: int) -> list[int]:
    """The base subset from path, or one rebuilt from VLLM_GGUF_TOKEN_STATS counts."""
    from .token_stats import TOKEN_STATS_ENV, load_token_counts

    base = load_ranked_ids(path)
    stats = os.environ.get(TOKEN_STATS_ENV)
    counts = load_token_counts(stats) if stats else None
    if counts is None or counts.sum() < MIN_COUNTED_TOKENS:
        if stats:
            logger.info(
                "Draft vocabulary: %s tokens counted so far, using %s",
                0 if counts is None else int(counts.sum()),
                path,
            )
        return base
    counts = counts[:vocab_size]
    rebuilt = build_draft_subset(base, counts, len(base))
    logger.info(
        "Draft vocabulary rebuilt from %d counted tokens: real-use coverage "
        "%.2f%% (base list %.2f%%)",
        int(counts.sum()),
        100 * _coverage(rebuilt, counts),
        100 * _coverage(base, counts),
    )
    try:
        with open(f"{stats}.draft_vocab.json", "w") as f:
            json.dump(rebuilt, f)
    except OSError:
        pass
    return rebuilt


def prune_draft_vocab(model: torch.nn.Module, token_ids: torch.Tensor) -> None:
    """Give a draft model a lm_head limited to token_ids."""
    full_head = model.lm_head
    vocab_size = full_head.weight.shape[0]
    head = PrunedLMHead(full_head, token_ids)
    # draft argmax index k -> target id k + d2t[k] (see LocalArgmaxMixin)
    d2t = head.token_ids - torch.arange(len(token_ids), device=head.token_ids.device)
    model.lm_head = head  # the target model keeps its own (possibly shared) head
    model.draft_id_to_target_id = d2t

    # Paths that ask for full-vocabulary logits still get correctly indexed ones.
    def compute_logits(hidden_states, *args, **kwargs):
        sub = model.logits_processor._apply_head(head, hidden_states, None)
        full = sub.new_full((sub.shape[0], vocab_size), float("-inf"))
        full[:, head.token_ids] = sub
        return full

    model.compute_logits = compute_logits


def _prune_after_load(speculator, path: str) -> None:
    """Shared by the V1 proposer and the V2 (Model Runner V2) speculator hooks."""
    model = speculator.model
    if not getattr(speculator, "use_local_argmax_reduction", False):
        logger.warning(
            "%s is set but use_local_argmax_reduction is off in "
            "--speculative-config; drafting over the full vocabulary.",
            DRAFT_VOCAB_ENV,
        )
        return
    if not hasattr(model, "get_top_tokens") or not hasattr(model, "lm_head"):
        logger.warning("%s: draft model has no local argmax; skipped.", DRAFT_VOCAB_ENV)
        return
    vocab_size = model.lm_head.weight.shape[0]
    ids = sorted(set(select_draft_vocab(path, vocab_size)))
    if not ids or ids[0] < 0 or ids[-1] >= vocab_size:
        raise ValueError(f"{path}: token ids must be in [0, {vocab_size})")
    token_ids = torch.tensor(ids, dtype=torch.long)
    prune_draft_vocab(model, token_ids)
    logger.info("Drafting over %d of %d tokens (%s)", len(token_ids), vocab_size, path)


def _wrap_load_model(cls, path: str) -> None:
    if getattr(cls, "_gguf_draft_vocab_patched", False):
        return
    original_load_model = cls.load_model

    @wraps(original_load_model)
    def load_model(self, target_model):
        original_load_model(self, target_model)
        _prune_after_load(self, path)

    cls.load_model = load_model
    cls._gguf_draft_vocab_patched = True


def _patch_proposer_load_model() -> None:
    path = os.environ.get(DRAFT_VOCAB_ENV)
    if not path:
        return
    # V1 model runner (EagleProposer and friends)
    from vllm.v1.spec_decode import llm_base_proposer

    _wrap_load_model(llm_base_proposer.SpecDecodeBaseProposer, path)
    # Model Runner V2 (the default in vLLM 0.32): MTP/EAGLE speculators
    try:
        from vllm.v1.worker.gpu.spec_decode import speculator as v2_speculator
    except ImportError:
        return
    _wrap_load_model(v2_speculator.DraftModelSpeculator, path)
