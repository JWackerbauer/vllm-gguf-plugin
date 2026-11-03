# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp (Qwen3.8-Flash-Next) GGUF adapter.

vLLM implements Qwen4Exp natively (vllm/models/qwen4_exp); this adapter maps a
llama.cpp ``qwen4exp`` GGUF onto it. The architecture is Qwen3.5's gated delta net
and MoE plus hyper-connections, QSA sparse attention (with an indexer) on the full
attention layers, and a per-layer n-gram embedding table (PLE) on one layer.

Verified against the Qwen/Qwen3.8-Flash-Next checkpoint (tools/qwen4exp_verify.py,
2026-11-03): every unquantized tensor matches after the transforms below; norms are
stored +1 (Gemma-style zero-centred, like Qwen3.5) except the linear-attention norm;
the expert router keeps the rows of the kept experts in their original order.

The PLE table (``per_layer_token_embd``, ~27 GB at IQ4_NL) is not part of the normal
weight stream: copying it through torch would need a second 27 GB in RAM. It is
filled directly from the GGUF file into vLLM's pinned host table after loading, see
quantization/qwen4exp_ple.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_reader import get_gguf_reader
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..quantization import qwen4exp_ple
from ..weight_utils import get_gguf_tensor_names, split_stacked_experts
from .base import GGUFWeight
from .qwen3_5 import Qwen35GGUFAdapter, _map_tensor_name, qwen35_layer_substr

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN4EXP_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")
QWEN4EXP_ARCHITECTURES = {
    "qwen4_exp": "Qwen4ExpForConditionalGeneration",
    "qwen4_exp_text": "Qwen4ExpForCausalLM",
}
GGUF_ARCH = "qwen4exp"
PLE_TABLE = "per_layer_token_embd.weight"
# the indexer's q and k projections are one tensor in HF; k is staged under this name
_INDEXER_K_STAGE = "self_attn.indexer.index_k_part."

# Applied before the Qwen3.5 rules (a WeightsMapper applies each matching substring
# once, in order): "hc_attn_norm." would otherwise hit Qwen3.5's "attn_norm.".
QWEN4EXP_SUBSTR: dict[str, str] = {
    "hc_attn_down.": "attn_hyper_connection.input_mix_weight_down.",
    "hc_attn_inject.": "attn_hyper_connection.block_inject_weight.",
    "hc_attn_up.": "attn_hyper_connection.input_mix_weight_up.",
    "hc_attn_norm.": "attn_hyper_connection.hc_norm.",
    "hc_ffn_down.": "mlp_hyper_connection.input_mix_weight_down.",
    "hc_ffn_inject.": "mlp_hyper_connection.block_inject_weight.",
    "hc_ffn_up.": "mlp_hyper_connection.input_mix_weight_up.",
    "hc_ffn_norm.": "mlp_hyper_connection.hc_norm.",
    "indexer.q_proj.": "self_attn.indexer.index_qk_proj.",
    "indexer.k_proj.": _INDEXER_K_STAGE,
    "indexer.q_norm.": "self_attn.indexer.q_layernorm.",
    "indexer.k_norm.": "self_attn.indexer.k_layernorm.",
    "ple_key.": "ple.key_proj.",
    "ple_value.": "ple.value_proj.",
    "ple_conv1d.": "ple.conv1d.",
    "ple_norm_key.": "ple.norm_key.",
    "ple_norm_query.": "ple.norm_query.",
    "ple_norm_conv.": "ple.norm_conv.",
}


def build_qwen4exp_mapper(is_multimodal: bool) -> WeightsMapper:
    backbone = "model.language_model." if is_multimodal else "model."
    return WeightsMapper(
        orig_to_new_prefix={
            "output_hc_down.": backbone + "hyper_connection_mixer.input_mix_weight_down.",
            "output_hc_up.": backbone + "hyper_connection_mixer.input_mix_weight_up.",
            "output_hc_norm.": backbone + "hyper_connection_mixer.hc_norm.",
            "token_embd.": backbone + "embed_tokens.",
            "blk.": backbone + "layers.",
            "output.": "lm_head.",
        },
        orig_to_new_substr=QWEN4EXP_SUBSTR | qwen35_layer_substr(is_moe=True),
    )


def _gguf_field(reader, key: str):
    field = reader.fields.get(key)
    return field.contents() if field is not None else None


class Qwen4ExpGGUFAdapter(Qwen35GGUFAdapter):
    """Qwen3.8-Flash-Next (llama.cpp ``qwen4exp``) GGUF adapter, text only."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN4EXP_MODEL_TYPES

    @classmethod
    def architecture(cls, config) -> str | None:
        return QWEN4EXP_ARCHITECTURES.get(config.model_type)

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        if files.mm_proj is not None:
            raise NotImplementedError("Qwen4Exp GGUF vision (mmproj) is not supported yet")
        patched = maybe_patch_hf_config_from_gguf(files.primary_backbone, hf_config)
        reader = get_gguf_reader(files.primary_backbone)
        text_config = patched.get_text_config()
        n_layer = _gguf_field(reader, f"{GGUF_ARCH}.block_count")
        if n_layer is not None and n_layer != text_config.num_hidden_layers:
            raise ValueError(
                f"GGUF has {n_layer} blocks, HF config {text_config.num_hidden_layers}"
            )
        # expert-pruned GGUFs keep fewer experts than the base config lists
        n_experts = _gguf_field(reader, f"{GGUF_ARCH}.expert_count")
        if n_experts is not None and n_experts != text_config.num_experts:
            logger.info(
                "Qwen4Exp GGUF keeps %d of %d experts per layer",
                n_experts, text_config.num_experts,
            )
            text_config.update({"num_experts": int(n_experts)})
        patched.architectures = [QWEN4EXP_ARCHITECTURES[patched.model_type]]
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        mapper = build_qwen4exp_mapper(is_multimodal=False)
        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            if name == PLE_TABLE or ".nextn." in name:
                continue  # PLE table: filled directly (qwen4exp_ple); MTP: separate file
            mapped = _map_tensor_name(mapper, name)
            if mapped is None:
                unmapped.append(name)
            else:
                name_map[name] = mapped
        if unmapped:
            logger.warning("No HF name for %d Qwen4Exp GGUF tensor(s), skipping: %s",
                           len(unmapped), unmapped)
        qwen4exp_ple.register_gguf_table(files.all_files, PLE_TABLE)
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        def merged() -> Iterable[GGUFWeight]:
            staged_q: dict[str, torch.Tensor] = {}
            staged_k: dict[str, torch.Tensor] = {}
            for name, weight in weights:
                # the indexer's q and k projections (BF16) are one [q; k] tensor in HF
                if ".self_attn.indexer.index_qk_proj.weight" in name:
                    staged_q[name] = weight
                elif _INDEXER_K_STAGE in name:
                    staged_k[name.replace(_INDEXER_K_STAGE, "self_attn.indexer.index_qk_proj.")] = weight
                else:
                    # PLE grouped norms are stored +1 like the other norms, but are not
                    # named "...norm.weight", which the Qwen3.5 rule keys on
                    if ".ple.norm_" in name and name.endswith(".weight"):
                        weight = weight - 1
                    yield name, weight
                    continue
                for key in list(staged_q):
                    if key in staged_k:
                        yield key, torch.cat([staged_q.pop(key), staged_k.pop(key)], dim=0)
            if staged_q or staged_k:
                raise ValueError(f"indexer q/k projection missing its other half: "
                                 f"{sorted(staged_q) + sorted(staged_k)}")

        # the Qwen3.5 rules (norms -1, GDN head order, A_log, conv1d, 1-D gates) then
        # apply to the renamed tensors; split_stacked_experts is done there too
        yield from super().transform_weights(merged(), model_config)
