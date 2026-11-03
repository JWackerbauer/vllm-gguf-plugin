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
from .. import qwen4exp_pp
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


def _unfuse_qsa_indexer() -> None:
    """Keep the QSA indexer projection out of ``qkv_proj`` for GGUF checkpoints.

    vLLM fuses q/k/v and the indexer's ``index_qk_proj`` into one linear unless LoRA
    is active. In a GGUF, q/k/v are quantized and the indexer is BF16, and the GGUF
    quant method needs one precision per fused layer. vLLM's LoRA layout (indexer as
    its own linear) is used instead: the model's packing table and the attention's
    ``fuse_indexer_projection`` follow it. Costs one small extra matmul per
    full-attention layer.
    """
    import copy

    from vllm.models.qwen4_exp.nvidia import model as q4_model
    from vllm.models.qwen4_exp.nvidia import qsa

    from ..quantization import GGUFConfig

    for cls in (q4_model.Qwen4ExpForCausalLM, q4_model.Qwen4ExpForConditionalGeneration):
        mapping = dict(cls.packed_modules_mapping)
        mapping["qkv_proj"] = [m for m in mapping["qkv_proj"] if "indexer" not in m]
        cls.packed_modules_mapping = mapping

    attn = qsa.Qwen4ExpQSAAttention
    if getattr(attn, "_gguf_unfused", False):
        return
    original_init = attn.__init__

    class _NoLoRA:  # truthy stand-in: the attention only checks "lora_config is None"
        pass

    def __init__(self, *args, vllm_config, **kwargs):
        if isinstance(vllm_config.quant_config, GGUFConfig) and vllm_config.lora_config is None:
            vllm_config = copy.copy(vllm_config)
            object.__setattr__(vllm_config, "lora_config", _NoLoRA())
        original_init(self, *args, vllm_config=vllm_config, **kwargs)

    attn.__init__ = __init__
    attn._gguf_unfused = True


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
        _unfuse_qsa_indexer()
        qwen4exp_pp.patch_model_state()
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


QWEN4EXP_MTP_MODEL_TYPES = ("qwen4_exp_mtp",)
# staging names for tensors the GGUF stores fused; split in transform_weights
_MTP_EH_STAGE = "mtp.fc_eh."
_DOWN_INJECT_STAGE = "hyper_connection.down_inject."
# vLLM builds the hyper-connection linears unquantized (BF16)
_HC_BF16_PARAMS = (
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
    "block_inject_weight.weight",
)


def build_qwen4exp_mtp_mapper(block_index: int) -> WeightsMapper:
    block = f"blk.{block_index}."
    return WeightsMapper(
        orig_to_new_prefix={
            f"{block}nextn.eh_proj.": _MTP_EH_STAGE,
            f"{block}nextn.enorm.": "mtp.pre_fc_norm_embedding.",
            f"{block}nextn.hnorm.": "mtp.pre_fc_norm_hidden.",
            f"{block}nextn.hc_head_down.": "mtp.hyper_connection_mixer.input_mix_weight_down.",
            f"{block}nextn.hc_head_up.": "mtp.hyper_connection_mixer.input_mix_weight_up.",
            f"{block}nextn.hc_head_norm.": "mtp.hyper_connection_mixer.hc_norm.",
            block: "mtp.layers.0.",
            "token_embd.": "embed_tokens.",
            "output.": "lm_head.",
        },
        orig_to_new_substr={
            # newer converters store the HC down projection and injection fused
            "hc_attn_down_inject.": "attn_" + _DOWN_INJECT_STAGE,
            "hc_ffn_down_inject.": "mlp_" + _DOWN_INJECT_STAGE,
        }
        | QWEN4EXP_SUBSTR
        | qwen35_layer_substr(is_moe=True),
    )


def _dequantize_rows(weight: torch.Tensor, weight_type: int) -> torch.Tensor:
    """Dequantize raw GGUF rows ([rows, bytes] uint8) to BF16 on the CPU."""
    import gguf

    values = gguf.quants.dequantize(weight.numpy(), gguf.GGMLQuantizationType(weight_type))
    return torch.from_numpy(values).to(torch.bfloat16)


def _keep_draft_gguf_quant_config() -> None:
    """Patches for the Qwen4Exp MTP drafter with a GGUF and pipeline parallelism.

    Qwen4ExpMTP derives a fresh draft quant config for its layer, which drops the
    loader's per-checkpoint state (unquantized modules such as the BF16 indexer):
    build the layer with the GGUF quant config the loader prepared.
    """
    from vllm.models.qwen4_exp.nvidia import mtp

    if getattr(mtp, "_gguf_draft_quant_patched", False):
        return
    original = mtp.get_draft_quant_config

    def get_draft_quant_config(vllm_config):
        from ..quantization import GGUFConfig

        if isinstance(vllm_config.quant_config, GGUFConfig):
            return vllm_config.quant_config
        return original(vllm_config)

    mtp.get_draft_quant_config = get_draft_quant_config
    mtp._gguf_draft_quant_patched = True

    # The drafter is one module on the last pipeline stage, but its forward embeds
    # the tokens only "if get_pp_group().is_first_rank" and otherwise waits for
    # intermediate tensors that no stage sends. Within the MTP module the drafter
    # is its own single stage. (is_last_rank, which picks the LM head, stays real.)
    real_get_pp_group = mtp.get_pp_group

    class _DrafterStage:
        def __init__(self, group):
            self._group = group

        is_first_rank = True

        def __getattr__(self, name):
            return getattr(self._group, name)

    mtp.get_pp_group = lambda: _DrafterStage(real_get_pp_group())


class Qwen4ExpMtpGGUFAdapter(Qwen4ExpGGUFAdapter):
    """MTP draft for Qwen3.8-Flash-Next from a GGUF holding one nextn block.

    Made for the base model's MTP head exported on its own (llama.cpp ``qwen4exp``
    with ``nextn_predict_layers`` 1, e.g. Qwen3.8-Flash-Next-Q4_K_M.gguf: the block,
    token_embd and output). With pipeline parallelism vLLM does not share the
    target's embedding with a drafter on the last stage, so the head's own
    embedding and LM head are loaded. Verified against the HF checkpoint
    (tools/qwen4exp_mtp_verify.py): eh_proj = [fc_embedding | fc_hidden] along the
    input dim, hc_*_down_inject = [down (hc_lowrank rows); inject (hc_count rows)],
    all norms stored +1.
    """

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN4EXP_MTP_MODEL_TYPES

    @classmethod
    def architecture(cls, config) -> str | None:
        return "Qwen4ExpMTP"

    def patch_hf_config(self, files: GGUFModelFiles, hf_config: PretrainedConfig) -> PretrainedConfig:
        reader = get_gguf_reader(files.primary_backbone)
        text_config = hf_config.get_text_config()
        n_experts = _gguf_field(reader, f"{GGUF_ARCH}.expert_count")
        if n_experts is not None and n_experts != text_config.num_experts:
            text_config.update({"num_experts": int(n_experts)})
        _unfuse_qsa_indexer()
        _keep_draft_gguf_quant_config()
        return hf_config

    def build_name_map(self, files: GGUFModelFiles, model_config: ModelConfig) -> dict[str, str]:
        reader = get_gguf_reader(files.primary_backbone)
        block_index = int(_gguf_field(reader, f"{GGUF_ARCH}.block_count")) - 1
        if not any(f"blk.{block_index}.nextn." in n for n in get_gguf_tensor_names(files.all_files)):
            raise RuntimeError(f"No nextn block {block_index} in {files.primary_backbone}")
        mapper = build_qwen4exp_mtp_mapper(block_index)
        name_map, unmapped = {}, []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            mapped = _map_tensor_name(mapper, name)
            if mapped is None:
                unmapped.append(name)
            else:
                name_map[name] = mapped
        if unmapped:
            logger.warning("No HF name for %d Qwen4Exp MTP tensor(s), skipping: %s",
                           len(unmapped), unmapped)
        # vLLM numbers the draft layer after the trunk (module prefix
        # ...layers.<num_hidden_layers>), while its weights load as layers.0; the
        # quant config matches unquantized modules by substring of the prefix. The
        # only BF16 linear in the block is the QSA indexer projection.
        n = model_config.hf_config.get_text_config().num_hidden_layers
        self.extra_unquantized_modules = (f"layers.{n}.self_attn.indexer.index_qk_proj",)
        logger.info("Loading Qwen4Exp MTP draft from GGUF block %d", block_index)
        return name_map

    def transform_weights(self, weights: Iterable[GGUFWeight], model_config: ModelConfig) -> Iterable[GGUFWeight]:
        text_config = model_config.hf_config.get_text_config()
        lowrank = int(text_config.hc_lowrank)

        def mtp_parts() -> Iterable[GGUFWeight]:
            types: dict[str, int] = {}
            for name, weight in weights:
                if name.endswith(".weight_type"):
                    # the iterator names it name.replace("weight", "weight_type"), which
                    # also hits names like input_mix_weight_up.weight
                    types[name.replace("weight_type", "weight")] = int(weight)
                    if name.startswith(_MTP_EH_STAGE):
                        for part in ("fc_embedding", "fc_hidden"):
                            yield name.replace(_MTP_EH_STAGE, f"mtp.{part}."), weight
                    continue  # re-emitted (or dropped, for BF16 targets) below
                qtype = types.get(name)
                if name.startswith(_MTP_EH_STAGE):
                    # [out, in_embedding + in_hidden]; for quantized rows the split
                    # point (2560 values) falls on a block boundary
                    half = weight.shape[1] // 2
                    if qtype is not None:
                        import gguf

                        block, _ = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qtype)]
                        assert text_config.hidden_size % block == 0
                    for part, chunk in (("fc_embedding", weight[:, :half]), ("fc_hidden", weight[:, half:])):
                        yield name.replace(_MTP_EH_STAGE, f"mtp.{part}."), chunk.contiguous()
                    continue
                if qtype is not None and name.endswith(_HC_BF16_PARAMS + (_DOWN_INJECT_STAGE + "weight",)):
                    weight, qtype = _dequantize_rows(weight, qtype), None
                if _DOWN_INJECT_STAGE in name:
                    assert qtype is None
                    base = name.removesuffix(_DOWN_INJECT_STAGE + "weight")
                    yield base + "hyper_connection.input_mix_weight_down.weight", weight[:lowrank].contiguous()
                    yield base + "hyper_connection.block_inject_weight.weight", weight[lowrank:].contiguous()
                    continue
                if qtype is not None:
                    yield name.replace("weight", "weight_type"), torch.tensor(qtype)
                if name.startswith("mtp.pre_fc_norm_"):
                    weight = weight - 1
                yield name, weight

        yield from super().transform_weights(mtp_parts(), model_config)
