# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np

import vllm_gguf_plugin.draft_vocab as draft_vocab
import vllm_gguf_plugin.token_stats as token_stats


def test_counter_accumulates_and_survives_restart(tmp_path):
    path = str(tmp_path / "counts.npy")
    counter = token_stats.TokenCounter(path)
    counter.add([5, 5, 7])
    counter.add([300_000])  # beyond the initial size: grows
    counter.flush()

    restarted = token_stats.TokenCounter(path)
    assert restarted.counts[5] == 2 and restarted.counts[7] == 1
    assert restarted.counts[300_000] == 1
    restarted.add([5])
    restarted.flush()
    assert token_stats.load_token_counts(path)[5] == 3


def test_subset_prefers_real_use_then_base_ranking():
    counts = np.zeros(20, np.int64)
    counts[[9, 4, 17]] = [50, 10, 1]  # 17 seen only once: not trusted yet
    base_ranked = [1, 2, 3, 4, 5, 6]
    subset = draft_vocab.build_draft_subset(base_ranked, counts, k=4)
    assert subset == [1, 2, 4, 9]  # 9 and 4 by count, then 1, 2 from the base


def test_selection_waits_for_enough_counts(tmp_path, monkeypatch):
    base = tmp_path / "base.json"
    base.write_text(json.dumps([3, 1, 2]))
    stats = tmp_path / "counts.npy"
    monkeypatch.setenv(token_stats.TOKEN_STATS_ENV, str(stats))
    monkeypatch.setattr(draft_vocab, "MIN_COUNTED_TOKENS", 100)

    counts = np.zeros(10, np.int64)
    counts[7] = 50
    np.save(stats, counts)
    assert draft_vocab.select_draft_vocab(str(base), 10) == [3, 1, 2]  # too few

    counts[7] = 500
    np.save(stats, counts)
    # k stays 3: the counted token 7 first, then the base ranking (3, 1)
    assert draft_vocab.select_draft_vocab(str(base), 10) == [1, 3, 7]
    assert json.loads((tmp_path / "counts.npy.draft_vocab.json").read_text()) == [
        1,
        3,
        7,
    ]


def test_output_processor_patch_counts_and_passes_through(tmp_path, monkeypatch):
    from vllm.v1.engine import output_processor

    cls = output_processor.OutputProcessor
    monkeypatch.setattr(cls, "process_outputs", lambda self, outs, *a, **k: "orig")
    monkeypatch.setattr(cls, "_gguf_token_stats_patched", False, raising=False)
    monkeypatch.setattr(token_stats, "_counter", None)
    path = str(tmp_path / "counts.npy")
    monkeypatch.setenv(token_stats.TOKEN_STATS_ENV, path)

    token_stats._patch_output_processor()
    outs = [SimpleNamespace(new_token_ids=[1, 2, 2]), SimpleNamespace(new_token_ids=[])]
    assert cls.process_outputs(None, outs) == "orig"
    token_stats._counter.flush()
    counts = token_stats.load_token_counts(path)
    assert counts[1] == 1 and counts[2] == 2
