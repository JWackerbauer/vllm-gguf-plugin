# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from functools import partial

import torch
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
)
from .utils import DEQUANT_TYPES, MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES, logger


_MAX_GRID_Z = 65535
# Types without MMQ kernels (i-quants) run batches of at least this many tokens as
# dequantize + Triton BF16 grouped GEMM, a chunk of experts at a time; the per-token
# vec kernels are ~10x slower there. The chunk bounds the BF16 copy (32 experts of
# Qwen3.8-Flash-Next: 315 MB).
_DEQUANT_MIN_TOKENS = int(os.environ.get("VLLM_GGUF_MOE_DEQUANT_MIN_TOKENS", "64"))
_DEQUANT_EXPERT_CHUNK = int(os.environ.get("VLLM_GGUF_MOE_DEQUANT_EXPERT_CHUNK", "32"))
_expert_maps: dict = {}


def _chunk_expert_map(num_experts: int, start: int, end: int, device) -> torch.Tensor:
    key = (num_experts, start, end, device)
    emap = _expert_maps.get(key)
    if emap is None:
        emap = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        emap[start:end] = torch.arange(end - start, dtype=torch.int32, device=device)
        _expert_maps[key] = emap
    return emap


def _fused_moe_dequant(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: MoEActivation,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    E, N, _ = w1.shape
    K = x.shape[1]
    H = w2.shape[1]
    I = N // 2
    # -1 marks unused slots (padding): give them weight 0 and a valid id
    invalid = topk_ids < 0
    topk_weights = topk_weights.masked_fill(invalid, 0)
    topk_ids = topk_ids.clamp(min=0)
    out = torch.zeros_like(x)
    for start in range(0, E, _DEQUANT_EXPERT_CHUNK):
        end = min(start + _DEQUANT_EXPERT_CHUNK, E)
        n = end - start
        w13 = ops.ggml_dequantize(
            w1[start:end].reshape(n * N, -1), weight_type, K, n * N, x.dtype
        ).view(n, N, K)
        w2d = ops.ggml_dequantize(
            w2[start:end].reshape(n * H, -1), weight_type2, I, n * H, x.dtype
        ).view(n, H, I)
        out += fused_experts(
            x, w13, w2d, topk_weights, topk_ids,
            activation=activation,
            global_num_experts=E,
            expert_map=_chunk_expert_map(E, start, end, x.device),
        )
    return out


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(activation_enum, out, inp)
        return out

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    if (
        weight_type2 in MMQ_QUANT_TYPES
        and weight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        block_size = ops.ggml_moe_get_block_size(weight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    elif (
        x.shape[0] >= _DEQUANT_MIN_TOKENS
        and weight_type in DEQUANT_TYPES
        and weight_type2 in DEQUANT_TYPES
    ):
        out_hidden_states = _fused_moe_dequant(
            x, w1, w2, topk_weights, topk_ids, weight_type, weight_type2,
            activation_enum,
        )
    elif weight_type2 in MMVQ_QUANT_TYPES and weight_type in MMVQ_QUANT_TYPES:
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        # the vec kernels launch one grid.z block per (token, expert) pair, and
        # grid.z is capped at 65535: 8192 tokens x top-10 would not launch
        chunk = _MAX_GRID_Z // top_k
        # The vec kernels index the expert weights with topk_ids directly. vLLM
        # marks unused slots with -1 (padding tokens, e.g. the CUDA graph
        # warm-up batch), which reads before W: route those to expert 0 and drop
        # their output. No host sync, so it stays CUDA-graph safe.
        invalid = topk_ids < 0
        topk_ids = topk_ids.clamp(min=0)
        for start in range(0, x.shape[0], chunk):
            end = min(start + chunk, x.shape[0])
            num_tokens = end - start
            ids = topk_ids[start:end]
            out = ops.ggml_moe_a8_vec(x[start:end], w1, ids, top_k, weight_type, N, num_tokens)
            out = act(out)

            out = ops.ggml_moe_a8_vec(
                out, w2, ids, 1, weight_type2, w2.shape[1], num_tokens * top_k
            )
            out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
                topk_weights[start:end].view(num_tokens, top_k, 1)
            )
            out.masked_fill_(invalid[start:end].view(num_tokens, top_k, 1), 0)
            ops.moe_sum(out, out_hidden_states[start:end])
    else:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                out = fused_mul_mat_gguf_op(inp, w1[ii], weight_type)
                out = act(out)
                current_state = fused_mul_mat_gguf_op(out, w2[ii], weight_type2).mul_(
                    ww
                )
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    del w1, w2, topk_weights, topk_ids, weight_type, weight_type2, activation
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w13_weight", w13_weight)

        w13_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_weight_type, extra_weight_attrs)
        layer.register_parameter("w13_weight_type", w13_weight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)

        w2_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_weight_type, extra_weight_attrs)
        layer.register_parameter("w2_weight_type", w2_weight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op

        return fused_moe_gguf_op(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            layer.w13_weight_type.weight_type,
            layer.w2_weight_type.weight_type,
            layer.activation.value,
        )
