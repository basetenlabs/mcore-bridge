# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import megatron.core
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import weakref
from contextlib import contextmanager, nullcontext
from importlib import metadata
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import (TEColumnParallelGroupedLinear, TEColumnParallelLinear,
                                                         TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear,
                                                         TERowParallelGroupedLinear, TERowParallelLinear)
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.parallel_state import get_expert_tensor_parallel_world_size, get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.router import TopKRouter
from packaging import version
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.other import transpose
from transformers.utils import is_torch_npu_available
from typing import Any, List, Optional, Tuple

from mcore_bridge.utils import get_current_device

from .utils import tuners_sharded_state_dict

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
mcore_016 = version.parse(megatron.core.__version__) >= version.parse('0.16.0rc0')


def apply_routed_lora(
    result: torch.Tensor,
    x: torch.Tensor,
    adapter_indices: torch.Tensor,
    lora_A_by_idx: dict,
    lora_B_by_idx: dict,
    scaling_by_idx: dict,
    dropout_by_idx: dict,
) -> torch.Tensor:
    """Apply per-token LoRA routing to result in-place.

    Args:
        result: [tokens, out_features] base output, updated in-place.
        x: [tokens, in_features] input to lora_A projections.
        adapter_indices: [tokens] LongTensor; 0 = no LoRA for that token.
        lora_A_by_idx: adapter_index -> lora_A callable.
        lora_B_by_idx: adapter_index -> lora_B callable.
        scaling_by_idx: adapter_index -> float scaling factor.
        dropout_by_idx: adapter_index -> dropout callable.
    Returns:
        result (same tensor, modified in-place).
    """
    for idx in adapter_indices.unique().tolist():
        if idx == 0:
            continue
        if idx not in lora_A_by_idx:
            continue
        mask = adapter_indices == idx
        rows = mask.nonzero(as_tuple=True)[0]
        x_sub = x.index_select(0, rows)
        a_out = lora_A_by_idx[idx](dropout_by_idx[idx](x_sub))
        if isinstance(a_out, tuple):
            a_out = a_out[0]
        b_out = lora_B_by_idx[idx](a_out)
        if isinstance(b_out, tuple):
            b_out = b_out[0]
        result.index_add_(0, rows, (b_out * scaling_by_idx[idx]).to(result.dtype))
    return result
MINDSPEED_015 = version.parse('0.15.0')


def _get_mindspeed_version():
    try:
        return version.parse(metadata.version('mindspeed'))
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _use_legacy_npu_local_linear() -> bool:
    if not is_torch_npu_available():
        return False
    mindspeed_version = _get_mindspeed_version()
    if mindspeed_version is None:
        # Fall back to the conservative path when the version is unknown so we
        # do not force an older NPU stack onto the 0.15 TE semantics.
        return True
    return mindspeed_version < MINDSPEED_015


def _build_local_te_linear(input_size: int, output_size: int, bias: bool, **kwargs):
    if _use_legacy_npu_local_linear():
        return nn.Linear(
            in_features=input_size,
            out_features=output_size,
            bias=bias,
        )
    local_kwargs = dict(kwargs)
    parallel_mode = None
    if is_torch_npu_available():
        # Local TE linear layers in MindSpeed 0.15.x use duplicated semantics,
        # and this path does not accept tp_group.
        local_kwargs.pop('tp_group', None)
        parallel_mode = 'duplicated'
    return TELinear(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        parallel_mode=parallel_mode,
        skip_weight_param_allocation=False,
        **local_kwargs,
    )


def _get_tensor_parallel_group_for_lora(base_layer):
    """Resolve the tensor-parallel group across TE and MindSpeed TE variants.

    Megatron's TE layers expose ``tp_group`` directly, but MindSpeed 0.15.x
    replaces some TE classes (for example
    ``MindSpeedTELayerNormColumnParallelLinear``) with implementations that keep
    the same tensor-parallel semantics under ``parallel_group`` instead. LoRA
    still needs to forward the right group into the newly created parallel
    adapter layers, otherwise adapter injection fails before training starts.
    """
    tp_group = getattr(base_layer, 'tp_group', None)
    if tp_group is not None:
        return tp_group
    return getattr(base_layer, 'parallel_group', None)


class LoraParallelLinear(MegatronModule, LoraLayer):

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ):
        config = base_layer.config
        super().__init__(config=config)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            LoraLayer.__init__(self, base_layer=base_layer)

        if use_dora:
            raise ValueError(f'{self.__class__.__name__} does not support DoRA yet, please set it to False')

        self.is_parallel_a = isinstance(base_layer, (TERowParallelLinear, TERowParallelGroupedLinear))
        self.is_grouped = isinstance(base_layer, TEGroupedLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.is_expert = getattr(base_layer, 'is_expert', False)
        self.sequence_parallel = getattr(base_layer, 'sequence_parallel', False)
        if self.is_expert:
            self.tp_size = get_expert_tensor_parallel_world_size()
            if self.tp_size > 1:
                raise ValueError('Currently, LoRA does not support ETP.')  # TODO: init/all-reduce
        else:
            self.tp_size = get_tensor_model_parallel_world_size()
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
        )

        self.is_target_conv_1d_layer = False
        # Set by _patch_lora_model after PEFT wrapping; used for routing.
        self._root_ref = None
        # Maps adapter slot index (int) -> PEFT adapter name (str); set by GPTBridge registry.
        self._idx_to_name: dict = {}

    def _unload_adapter(self, name: str) -> None:
        """Remove all state for adapter `name` from this layer."""
        for d in (self.lora_A, self.lora_B, self.lora_dropout, self.scaling, self.r, self.lora_alpha):
            d.pop(name, None)
        if hasattr(self, 'lora_bias'):
            self.lora_bias.pop(name, None)
        remaining = list(self.lora_A.keys())
        if remaining:
            self.set_adapter(remaining)

    def update_layer(self, adapter_name, r, *, lora_alpha, lora_dropout, init_lora_weights, use_rslora, lora_bias,
                     **kwargs):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        # lora needs to be forced to upgrade to 32-bit precision, otherwise it will overflow
        kwargs = {
            'skip_bias_add': False,
            'init_method': self.config.init_method,
            'config': self.config,
            'is_expert': self.is_expert,
        }
        if mcore_013 and not (mcore_016 and self.is_grouped):
            tp_group = _get_tensor_parallel_group_for_lora(self.base_layer)
            if tp_group is not None:
                kwargs['tp_group'] = tp_group
        if isinstance(self.base_layer, TopKRouter):
            router_shape = self.base_layer.weight.shape
            lora_a = _build_local_te_linear(router_shape[1], r, lora_bias, **kwargs)
            lora_b = _build_local_te_linear(r, router_shape[0], lora_bias, **kwargs)
        elif self.is_parallel_a:
            in_features = self.in_features * self.tp_size
            if self.is_grouped:
                lora_a = TERowParallelGroupedLinear(
                    num_gemms=self.base_layer.num_gemms,
                    input_size=in_features,
                    output_size=r,
                    bias=False,
                    **kwargs,
                )
                lora_b = TEGroupedLinear(
                    num_gemms=self.base_layer.num_gemms,
                    input_size=r,
                    output_size=self.out_features,
                    bias=lora_bias,
                    parallel_mode=None,
                    **kwargs,
                )
            else:
                lora_a = TERowParallelLinear(
                    input_size=in_features,
                    output_size=r,
                    bias=False,
                    input_is_parallel=True,
                    **kwargs,
                )
                lora_b = _build_local_te_linear(r, self.out_features, lora_bias, **kwargs)
                lora_a.parallel_mode = self.base_layer.parallel_mode  # fix moe_shared_expert_overlap
        else:
            if is_torch_npu_available():
                out_features = self.out_features
            else:
                out_features = self.out_features * self.tp_size
            if self.is_grouped:
                lora_a = TEGroupedLinear(
                    num_gemms=self.base_layer.num_gemms,
                    input_size=self.in_features,
                    output_size=r,
                    bias=lora_bias,
                    parallel_mode=None,
                    **kwargs)
                lora_b = TEColumnParallelGroupedLinear(
                    num_gemms=self.base_layer.num_gemms,
                    input_size=r,
                    output_size=out_features,
                    bias=lora_bias,
                    **kwargs,
                )
            else:
                lora_a = _build_local_te_linear(self.in_features, r, lora_bias, **kwargs)
                lora_b = TEColumnParallelLinear(
                    input_size=r,
                    output_size=out_features,
                    bias=lora_bias,
                    gather_output=False,
                    **kwargs,
                )
                lora_b.parallel_mode = self.base_layer.parallel_mode  # fix moe_shared_expert_overlap
        for lora in [lora_a, lora_b]:
            if getattr(lora, 'parallel_mode', None) is None and hasattr(lora, 'weight'):  # TODO: experts
                if isinstance(self.base_layer, TopKRouter):
                    sequence_parallel = self.base_layer.weight.sequence_parallel
                else:
                    sequence_parallel = self.sequence_parallel
                lora.weight.sequence_parallel = sequence_parallel
        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b
        if hasattr(self, 'lora_bias'):
            self.lora_bias[adapter_name] = lora_bias
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / (r**0.5)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _get_rng_context(self, lora):
        if not get_cuda_rng_tracker().is_initialized():
            return nullcontext()
        if self.is_expert:
            rng_context = get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name())
        elif getattr(lora, 'parallel_mode', None) is None:
            rng_context = nullcontext()
        else:
            rng_context = get_cuda_rng_tracker().fork()
        return rng_context

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            lora_a = self.lora_A[adapter_name]
            lora_b = self.lora_B[adapter_name]
            if isinstance(lora_a, TEGroupedLinear):
                weights_a = [getattr(lora_a, f'weight{i}') for i in range(lora_a.num_gemms)]
            else:
                weights_a = [lora_a.weight]
            if isinstance(lora_b, TEGroupedLinear):
                weights_b = [getattr(lora_b, f'weight{i}') for i in range(lora_b.num_gemms)]
            else:
                weights_b = [lora_b.weight]
            with self._get_rng_context(lora_a):
                for weight_a in weights_a:
                    if init_lora_weights is True:
                        # initialize A the same way as the default for nn.Linear and B to zero
                        # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                        nn.init.kaiming_uniform_(weight_a, a=math.sqrt(5))
                    elif init_lora_weights.lower() == 'gaussian':
                        nn.init.normal_(weight_a, std=1 / self.r[adapter_name])
                    else:
                        raise ValueError(f'Unknown initialization {init_lora_weights=}')
            for weight_b in weights_b:
                nn.init.zeros_(weight_b)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    @contextmanager
    def _patch_router_gating(self):
        origin_gating = self.base_layer.__class__.gating
        _lpl = self  # capture LoraParallelLinear for use inside closure

        def gating(_self, x):
            result = origin_gating(_self, x)

            _root = _lpl._root_ref() if _lpl._root_ref is not None else None
            _adapter_indices = getattr(_root, '_lora_adapter_indices', None) if _root is not None else None

            if _adapter_indices is not None and _lpl._idx_to_name:
                # Per-sequence routing: dispatch LoRA delta per adapter slot
                out_feat = result.shape[-1]
                total_tokens = result.numel() // out_feat
                batch = _adapter_indices.shape[0]
                if total_tokens % batch == 0:
                    seq = total_tokens // batch
                    token_idx = _adapter_indices.unsqueeze(0).expand(seq, batch).reshape(-1)
                else:
                    token_idx = _adapter_indices.reshape(-1)
                token_idx = token_idx.to(device=x.device)
                result_flat = result.reshape(total_tokens, out_feat)
                x_flat = x.reshape(total_tokens, -1)
                for idx, slot in _lpl._idx_to_name.items():
                    if slot not in _lpl.lora_A:
                        continue
                    mask = token_idx == idx
                    if not mask.any():
                        continue
                    rows = mask.nonzero(as_tuple=True)[0]
                    x_sub = x_flat.index_select(0, rows).to(result.dtype)
                    delta = F.linear(_lpl.lora_dropout[slot](x_sub),
                                     _lpl.lora_A[slot].weight.to(result.dtype))
                    delta = F.linear(delta, _lpl.lora_B[slot].weight.to(result.dtype))
                    result_flat.index_add_(0, rows, (delta * _lpl.scaling[slot]).to(result.dtype))
                result = result_flat.view_as(result)
            else:
                # Single-adapter path
                for active_adapter in _lpl.active_adapters:
                    if active_adapter not in _lpl.lora_A.keys():
                        continue
                    lora_A = _lpl.lora_A[active_adapter]
                    lora_B = _lpl.lora_B[active_adapter]
                    dropout = _lpl.lora_dropout[active_adapter]
                    scaling = _lpl.scaling[active_adapter]
                    x = x.to(result.dtype)
                    lora_result = F.linear(dropout(x), lora_A.weight.to(result.dtype))
                    if isinstance(lora_result, tuple):
                        lora_result = lora_result[0]
                    lora_result = F.linear(lora_result, lora_B.weight.to(result.dtype))
                    if isinstance(lora_result, tuple):
                        lora_result = lora_result[0]
                    result = result + lora_result * scaling
            return result

        self.base_layer.__class__.gating = gating
        try:
            yield
        finally:
            self.base_layer.__class__.gating = origin_gating

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype
        if self.disable_adapters and self.merged:
            self.unmerge()

        if isinstance(self.base_layer, TELayerNormColumnParallelLinear):
            if self.disable_adapters or self.merged:
                self.base_layer.return_layernorm_output = False
                result, bias = self.base_layer(x, *args, **kwargs)
            else:
                self.base_layer.return_layernorm_output = True
                if is_torch_npu_available():
                    # NPU: base_layer only returns (output, bias); it does not expose the LayerNorm/RMSNorm output.
                    inp = x  # Keep the original pre-norm input.
                    result, bias = self.base_layer(inp, *args, **kwargs)

                    # Key: For LoRA we need the same "x" as in the non-NPU branch, i.e. the post-norm activation
                    # (LayerNorm/RMSNorm output, which is the actual input to the fused linear).
                    if hasattr(self.base_layer, 'config') and (hasattr(self.base_layer, '_layernorm')
                                                               or hasattr(self.base_layer, '_rmsnorm')):
                        norm_type = getattr(self.base_layer.config, 'normalization', None)

                        if norm_type == 'LayerNorm':
                            if not hasattr(self.base_layer, '_layernorm'):
                                raise RuntimeError(
                                    'NPU LoRA path expects base_layer to provide `_layernorm`, but it is missing. '
                                    'Cannot reconstruct the post-LayerNorm activation for LoRA.')
                            x = self.base_layer._layernorm(inp)
                        else:
                            # Default to RMSNorm path when normalization is not LayerNorm.
                            if not hasattr(self.base_layer, '_rmsnorm'):
                                raise RuntimeError(
                                    'NPU LoRA path expects base_layer to provide `_rmsnorm`, but it is missing. '
                                    'Cannot reconstruct the post-RMSNorm activation for LoRA.')
                            x = self.base_layer._rmsnorm(inp)
                    else:
                        raise RuntimeError('NPU LoRA path requires base_layer to expose post-norm activations '
                                           '(LayerNorm/RMSNorm output). Expected base_layer to have `config` '
                                           'and either `_layernorm` or `_rmsnorm`. '
                                           f'Got base_layer type: {type(self.base_layer)}. ')
                else:
                    (result, x), bias = self.base_layer(x, *args, **kwargs)
        elif isinstance(self.base_layer, (TELinear, TEGroupedLinear)):
            result, bias = self.base_layer(x, *args, **kwargs)
        elif isinstance(self.base_layer, TopKRouter):
            with self._patch_router_gating():
                result, bias = self.base_layer(x, *args, **kwargs)
        else:
            raise ValueError(f'Unsupported base layer type: {type(self.base_layer)}')
        if not isinstance(self.base_layer, TopKRouter) and not self.disable_adapters and not self.merged:
            _root = self._root_ref() if self._root_ref is not None else None
            _adapter_indices = getattr(_root, '_lora_adapter_indices', None) if _root is not None else None

            # Determine whether to use the routing path and which token_indices to use.
            # Expert layers (is_expert=True) need post-dispatch indices from RouterReplay
            # (_expert_lora_indices set by _patch_MoELayer in patcher.py). Non-expert layers
            # expand the per-sequence adapter_indices to per-token indices directly.
            token_indices = None
            use_routing = _adapter_indices is not None and self._idx_to_name
            if use_routing:
                if self.is_expert:
                    # RouterReplay path: post-dispatch token-level indices set by MoELayer patch
                    token_indices = getattr(_root, '_expert_lora_indices', None)
                    if token_indices is not None:
                        token_indices = token_indices.to(device=x.device)
                    else:
                        use_routing = False  # no RouterReplay → fall to single-adapter path
                else:
                    # Standard expansion: [batch] → [total_tokens]
                    out_feat = result.shape[-1]
                    total_tokens = result.numel() // out_feat
                    batch = _adapter_indices.shape[0]
                    if total_tokens % batch == 0:
                        seq = total_tokens // batch
                        token_indices = (
                            _adapter_indices.unsqueeze(0).expand(seq, batch).reshape(-1)
                        )
                    else:
                        token_indices = _adapter_indices.reshape(-1)
                    token_indices = token_indices.to(device=x.device)

            if use_routing and token_indices is not None:
                # ── routing path: per-token adapter dispatch ──────────────────
                lora_A_by_idx, lora_B_by_idx, scaling_by_idx, dropout_by_idx = {}, {}, {}, {}
                for idx, slot in self._idx_to_name.items():
                    if slot in self.lora_A:
                        lora_A_by_idx[idx] = self.lora_A[slot]
                        lora_B_by_idx[idx] = self.lora_B[slot]
                        scaling_by_idx[idx] = self.scaling[slot]
                        dropout_by_idx[idx] = self.lora_dropout[slot]

                if lora_A_by_idx:
                    first_A = next(iter(lora_A_by_idx.values()))
                    dtype = first_A.weight0.dtype if isinstance(first_A, TEGroupedLinear) else first_A.weight.dtype

                    out_feat = result.shape[-1]
                    total_tokens = result.numel() // out_feat
                    x_flat = x.to(dtype).contiguous().reshape(total_tokens, -1)
                    # Use a fresh delta buffer so index_add_ has no autograd-view conflict.
                    # The out-of-place addition to result avoids in-place view issues.
                    delta = torch.zeros(total_tokens, out_feat, dtype=dtype, device=result.device)
                    apply_routed_lora(delta, x_flat, token_indices,
                                      lora_A_by_idx, lora_B_by_idx, scaling_by_idx, dropout_by_idx)
                    result = result + delta.view_as(result).to(result.dtype)
            else:
                # ── existing single-adapter path ──────────────────────────────
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    dtype = lora_A.weight0.dtype if isinstance(lora_A, TEGroupedLinear) else lora_A.weight.dtype
                    x = x.to(dtype)

                    lora_result = lora_A(dropout(x), *args, **kwargs) if isinstance(lora_A, TEGroupedLinear) else lora_A(
                        dropout(x))
                    if isinstance(lora_result, tuple):
                        lora_result = lora_result[0]
                    lora_result = lora_B(lora_result, *args, **kwargs) if isinstance(
                        lora_B, TEGroupedLinear) else lora_B(lora_result)
                    if isinstance(lora_result, tuple):
                        lora_result = lora_result[0]
                    lora_result = lora_result * scaling
                    result = result + lora_result

        result = result.to(previous_dtype)
        return result, bias

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        sharded_state_dict = tuners_sharded_state_dict(self, prefix, sharded_offsets, metadata)
        if prefix.endswith('linear_fc1.'):
            if isinstance(self.base_layer, TEGroupedLinear) and self.config.gated_linear_unit:
                num_global_experts = (parallel_state.get_expert_model_parallel_world_size() * self.base_layer.num_gemms)
                local_expert_indices_offset = (
                    parallel_state.get_expert_model_parallel_rank() * self.base_layer.num_gemms)
                ep_axis = len(sharded_offsets)
                for i in range(self.base_layer.num_gemms):
                    new_sharded_offsets = (
                        *sharded_offsets,
                        (ep_axis, local_expert_indices_offset + i, num_global_experts),
                    )
                    for k in (f'{prefix}base_layer.weight{i}', f'{prefix}base_layer.bias{i}'):
                        if k in sharded_state_dict:
                            sharded_state_dict[k] = apply_swiglu_sharded_factory(sharded_state_dict[k],
                                                                                 new_sharded_offsets)
            else:
                for k, v in sharded_state_dict.items():
                    if k in [f'{prefix}base_layer.weight', f'{prefix}base_layer.bias']:
                        sharded_state_dict[k] = apply_swiglu_sharded_factory(sharded_state_dict[k], sharded_offsets)
        return sharded_state_dict

    def get_delta_weights(self, adapter) -> List[torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        lora_A = self.lora_A[adapter]
        lora_B = self.lora_B[adapter]
        if self.is_grouped:
            weight_A = [getattr(lora_A, f'weight{i}') for i in range(lora_A.num_gemms)]
            weight_B = [getattr(lora_B, f'weight{i}') for i in range(lora_B.num_gemms)]
        else:
            weight_A = [self.lora_A[adapter].weight]
            weight_B = [self.lora_B[adapter].weight]
        output_tensor = []
        assert len(weight_A) == len(weight_B)
        for i in range(len(weight_B)):
            output_tensor.append(transpose(weight_B[i] @ weight_A[i], self.fan_in_fan_out) * self.scaling[adapter])

        return output_tensor

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        base_layer = self.get_base_layer()
        origin_device = base_layer.weight0.device if self.is_grouped else base_layer.weight.device
        if origin_device.type == 'cpu':
            self.to(device=get_current_device())
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                if self.is_grouped:
                    orig_weights = [getattr(base_layer, f'weight{i}') for i in range(base_layer.num_gemms)]
                else:
                    orig_weights = [base_layer.weight]
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = [weight.data.clone() for weight in orig_weights]
                    delta_weights = self.get_delta_weights(active_adapter)
                    for orig_weight, delta_weight in zip(orig_weights, delta_weights):
                        orig_weight += delta_weight
                    if not all(torch.isfinite(orig_weights[i]).all() for i in range(len(orig_weights))):
                        raise ValueError(
                            f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    if self.is_grouped:
                        for i in range(base_layer.num_gemms):
                            weight = getattr(base_layer, f'weight{i}')
                            weight.data = orig_weights[i]
                    else:
                        base_layer.weight.data = orig_weights[0]
                else:
                    delta_weights = self.get_delta_weights(active_adapter)
                    for orig_weight, delta_weight in zip(orig_weights, delta_weights):
                        orig_weight.data += delta_weight
                self.merged_adapters.append(active_adapter)
        if origin_device.type == 'cpu':
            self.to(device=origin_device)

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter weights from the base weights.

        This method reverses the merge operation by subtracting the LoRA delta weights
        from the base layer weights, restoring the original base weights.
        """
        if not self.merged:
            # No adapters to unmerge
            return

        base_layer = self.get_base_layer()
        origin_device = base_layer.weight0.device if self.is_grouped else base_layer.weight.device
        if origin_device.type == 'cpu':
            self.to(device=get_current_device())

        for active_adapter in self.merged_adapters:
            if active_adapter in self.lora_A.keys():
                if self.is_grouped:
                    orig_weights = [getattr(base_layer, f'weight{i}') for i in range(base_layer.num_gemms)]
                else:
                    orig_weights = [base_layer.weight]

                delta_weights = self.get_delta_weights(active_adapter)
                for orig_weight, delta_weight in zip(orig_weights, delta_weights):
                    # Subtract the delta weight to unmerge
                    orig_weight.data -= delta_weight

        # Clear the merged adapters list
        self.merged_adapters = []

        if origin_device.type == 'cpu':
            self.to(device=origin_device)
