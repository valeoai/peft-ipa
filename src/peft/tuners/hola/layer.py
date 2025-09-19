# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import warnings
import copy
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils.imports import is_xpu_available
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx, get_bnb_param_type
from peft.utils.other import transpose

from torch.nn.init import _calculate_correct_fan

from .config import HolaConfig
from .._buffer_dict import BufferDict

def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
) -> torch.Tensor:
    """
    Kaiming Uniform Initialisation adapted to accept a `torch.Generator` object for PRNG.

    Args:
        tensor_or_shape (`Union[torch.Tensor, tuple[int, ...]]`):
            Tensor to initialise, or shape of new tensor to create and then initialise.
        generator: (`torch.Generator`):
            Generator object that manages the state of the PRNG algorithm in use.

    Returns:
        `torch.Tensor`: The initialised tensor.
    """
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
    
def uniform_cubic_initialization(tensor_or_shape):
    """
    Initializes the given tensor with samples from the distribution
    f_X(x) = (3x^2) / (2a^3) for x in [-a, a].

    Args:
        tensor (torch.Tensor): The tensor to be initialized.
        a (float): The parameter defining the range [-a, a].

    Returns:
        torch.Tensor: The initialized tensor.
    """

    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    print(bound)

    # Step 1: Generate uniform random samples in [0, 1]
    u = torch.rand_like(tensor)

    # Step 2: Transform using the inverse CDF
    z = 2 * bound * u - bound

    # Handle the sign explicitly for the cubic root
    tensor.data = torch.sign(z) * torch.abs(z).pow(1 / 3)

    return tensor


class HolaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("hola_A", "hola_B", "hola_tensor", "hola_code", "hola_embedding_A", "hola_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "hola_alpha", "scaling", "hola_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.hola_alpha = {}
        self.scaling = {}
        self.size_code = {}
        self.hola_dropout = nn.ModuleDict({})

        self.hola_A = nn.ParameterDict({})
        self.hola_B = nn.ParameterDict({})
        self.hola_tensor = nn.ParameterDict({})
        self.hola_code = nn.ParameterDict({})
        # For Embedding layer
        self.hola_embedding_A = nn.ParameterDict({})
        self.hola_embedding_B = nn.ParameterDict({})
        self.hola_embedding_tensor = nn.ParameterDict({})
        self.hola_embedding_code = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, size_code, hola_alpha, hola_dropout, init_hola_weights
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.size_code[adapter_name] = size_code
        self.hola_alpha[adapter_name] = hola_alpha
        if hola_dropout > 0.0:
            hola_dropout_layer = nn.Dropout(p=hola_dropout)
        else:
            hola_dropout_layer = nn.Identity()

        self.hola_dropout.update(nn.ModuleDict({adapter_name: hola_dropout_layer}))
        # Actual trainable parameters

        if adapter_name == 'init':
            self.hola_A[adapter_name] = nn.Parameter(torch.rand(r, self.in_features))
            self.hola_B[adapter_name] = nn.Parameter(torch.rand(self.out_features, r))
            self.hola_tensor[adapter_name] = nn.Parameter(torch.rand(r, r, size_code))
            self.hola_code[adapter_name] = nn.Parameter(torch.zeros(size_code))
            self.reset_hola_parameters(adapter_name, init_hola_weights)
        else:
            assert 'init' in self.hola_A
            self.hola_A[adapter_name] = self.hola_A['init']
            self.hola_B[adapter_name] = self.hola_B['init']
            self.hola_tensor[adapter_name] =  self.hola_tensor['init']
            self.hola_code[adapter_name] = nn.Parameter(torch.zeros(size_code))

        self.scaling[adapter_name] = hola_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_hola_parameters(self, adapter_name, init_hola_weights):
        if init_hola_weights is None:
            return

        if adapter_name in self.hola_A.keys():
            if init_hola_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/Hola/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/holalib/layers.py#L124
                _kaiming_init(self.hola_A[adapter_name])#, a=math.sqrt(5))
                _kaiming_init(self.hola_B[adapter_name])#, a=math.sqrt(5))
                _kaiming_init(self.hola_tensor[adapter_name])#, a=math.sqrt(5))
                nn.init.zeros_(self.hola_code[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_hola_weights=}")
            
        if adapter_name in self.hola_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/Hola/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/holalib/layers.py#L59-L60
            nn.init.zeros_(self.hola_embedding_A[adapter_name])
            nn.init.normal_(self.hola_embedding_B[adapter_name])
    
    def set_hola_code(self, adapter_name, code=None):
        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        for a_n in adapter_name:
            if code is None:
                self.hola_code[a_n].zero_()
            else:         
                assert code.size() == self.hola_code[a_n].size()
                self.hola_code[a_n].copy_(code)

    def get_hola_code(self, adapter_name):
        return copy.deepcopy(self.hola_code[adapter_name].data)

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.hola_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.hola_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.hola_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.hola_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

# Below code is based on https://github.com/microsoft/Hola/blob/main/holalib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, HolaLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        size_code: int = 0,
        hola_alpha: int = 1,
        hola_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_hola_weights: Union[bool, str, torch.Tensor] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        HolaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            size_code=size_code,
            hola_alpha=hola_alpha,
            hola_dropout=hola_dropout,
            init_hola_weights=init_hola_weights,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

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

        for active_adapter in adapter_names:
            if active_adapter in self.hola_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.hola_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.hola_B[adapter].weight.device
        dtype = self.hola_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.hola_A[adapter].weight
        weight_B = self.hola_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) # * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.hola_A[adapter].weight.data = weight_A.to(dtype)
            self.hola_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.hola_A.keys():
                    continue
                hola_A = self.hola_A[active_adapter]
                hola_B = self.hola_B[active_adapter]
                hola_tensor = self.hola_tensor[active_adapter]
                hola_code = self.hola_code[active_adapter]
                dropout = self.hola_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(hola_A.dtype)

                hola_core = torch.einsum('ijk,k->ij', hola_tensor, hola_code)
                # hola_A_code = hola_core @ hola_A
                result = result + F.linear(F.linear(F.linear(dropout(x), hola_A), hola_core), hola_B)
                # result = result + hola_B(hola_A(dropout(x)) @ hola_core) # * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hola." + rep

class _ConvNd(nn.Module, HolaLayer):
    # Lora implemented in a conv(2,3)d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        hola_alpha: int = 1,
        hola_dropout: float = 0.0,
        init_hola_weights: Union[bool, str] = True,
        use_rshola: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        HolaLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()

        self.update_layer(
            adapter_name,
            r,
            hola_alpha=hola_alpha,
            hola_dropout=hola_dropout,
            init_hola_weights=init_hola_weights,
            use_rshola=use_rshola,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, hola_alpha, hola_dropout, init_hola_weights, use_rshola, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.hola_alpha[adapter_name] = hola_alpha
        if hola_dropout > 0.0:
            hola_dropout_layer = nn.Dropout(p=hola_dropout)
        else:
            hola_dropout_layer = nn.Identity()

        self.hola_dropout[adapter_name] = hola_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        conv_layer = type(base_layer)
        out_kernel = out_stride = (1,) * (self._kernel_dim - 2)
        self.hola_A[adapter_name] = conv_layer(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.hola_B[adapter_name] = conv_layer(r, self.out_features, out_kernel, out_stride, bias=False)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def _get_dora_factor_view(self):
        return (-1,) + (1,) * (self._kernel_dim - 1)

    def dora_init(self, adapter_name: str) -> None:
        if self.hola_magnitude_vector is None:
            # first dora layer being added, add hola_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("hola_magnitude_vector",)

        dora_layer_class = self._get_dora_layer_class()
        dora_layer = dora_layer_class(fan_in_fan_out=False)
        hola_A = self.hola_A[adapter_name].weight
        hola_B = self.hola_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), hola_A=hola_A, hola_B=hola_B, scaling=scaling)
        self.hola_magnitude_vector[adapter_name] = dora_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

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

        for active_adapter in adapter_names:
            if active_adapter in self.hola_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.hola_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.hola_magnitude_vector[active_adapter].weight / weight_norm
                        orig_weights = dora_factor.view(*self._get_dora_factor_view()) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.hola_magnitude_vector[active_adapter]
                            .get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.hola_magnitude_vector[active_adapter].weight / weight_norm
                        new_weight = dora_factor.view(*self._get_dora_factor_view()) * (
                            base_layer.weight.data + delta_weight
                        )
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.hola_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.hola_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(*self._get_dora_factor_view()) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.hola_B[adapter].weight.device
        dtype = self.hola_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.hola_A[adapter].weight
        weight_B = self.hola_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/hola.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            output_tensor = (
                self.conv_fn(
                    weight_A.transpose(0, 1),
                    weight_B,
                ).transpose(0, 1)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.hola_A[adapter].weight.data = weight_A.to(dtype)
            self.hola_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.hola_A.keys():
                    continue
                hola_A = self.hola_A[active_adapter]
                hola_B = self.hola_B[active_adapter]
                dropout = self.hola_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(hola_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + hola_B(hola_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.hola_magnitude_vector[active_adapter](
                        x,
                        hola_A=hola_A,
                        hola_B=hola_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hola." + rep


class Conv2d(_ConvNd):
    # Lora implemented in a conv2d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 4:
            raise ValueError(f"Conv2d layer kernel must have 4 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv2d

class Conv3d(_ConvNd):
    # Lora implemented in a conv3d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 5:
            raise ValueError(f"Conv3d layer kernel must have 5 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv3d

def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    hola_config: HolaConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(hola_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(hola_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        kwargs.update(hola_config.loftq_config)
        new_module = Conv3d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = hola_config.fan_in_fan_out = False
        kwargs.update(hola_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = hola_config.fan_in_fan_out = True
        kwargs.update(hola_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
