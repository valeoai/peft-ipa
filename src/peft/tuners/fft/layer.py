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

from .config import FFTConfig
from .._buffer_dict import BufferDict

class FFTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("fft", "fft_embedding_A", "fft_embedding_B")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.scaling = {}
        self.size_code = {}

        self.fft = nn.ParameterDict({})
        # For Embedding layer
        self.fft_embedding_A = nn.ParameterDict({})
        self.fft_embedding_B = nn.ParameterDict({})
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
        self, adapter_name, init_fft_weights
    ):
        self.fft[adapter_name] = nn.Parameter(torch.empty(self.out_features, self.in_features), requires_grad=True)
        self.reset_fft_parameters(adapter_name, init_fft_weights)
        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_fft_parameters(self, adapter_name, init_fft_weights):
        if init_fft_weights is None:
            return

        if adapter_name in self.fft.keys():
            if init_fft_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/FFT/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/fftlib/layers.py#L124
                nn.init.zeros_(self.fft[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_fft_weights=}")
            
        if adapter_name in self.fft_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/FFT/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/fftlib/layers.py#L59-L60
            nn.init.zeros_(self.fft_embedding_A[adapter_name])
            nn.init.normal_(self.fft_embedding_B[adapter_name])


    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

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

# Below code is based on https://github.com/microsoft/FFT/blob/main/fftlib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, FFTLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_fft_weights: Union[bool, str, torch.Tensor] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        FFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            init_fft_weights=init_fft_weights,
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
            if active_adapter in self.fft_A.keys():
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
            if active_adapter in self.fft_A.keys():
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
        device = self.fft_B[adapter].weight.device
        dtype = self.fft_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.fft_A[adapter].weight
        weight_B = self.fft_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) # * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.fft_A[adapter].weight.data = weight_A.to(dtype)
            self.fft_B[adapter].weight.data = weight_B.to(dtype)

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
                if active_adapter not in self.fft.keys():
                    continue
                fft = self.fft[active_adapter]
                x = x.to(fft.dtype)
                # fft_A_code = fft_core @ fft_A
                result = result + F.linear(x, fft)
                # result = result + fft_B(fft_A(dropout(x)) @ fft_core) # * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fft." + rep

class _ConvNd(nn.Module, FFTLayer):
    # Lora implemented in a conv(2,3)d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        fft_alpha: int = 1,
        fft_dropout: float = 0.0,
        init_fft_weights: Union[bool, str] = True,
        use_rsfft: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        FFTLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()

        self.update_layer(
            adapter_name,
            r,
            fft_alpha=fft_alpha,
            fft_dropout=fft_dropout,
            init_fft_weights=init_fft_weights,
            use_rsfft=use_rsfft,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, fft_alpha, fft_dropout, init_fft_weights, use_rsfft, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.fft_alpha[adapter_name] = fft_alpha
        if fft_dropout > 0.0:
            fft_dropout_layer = nn.Dropout(p=fft_dropout)
        else:
            fft_dropout_layer = nn.Identity()

        self.fft_dropout[adapter_name] = fft_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        conv_layer = type(base_layer)
        out_kernel = out_stride = (1,) * (self._kernel_dim - 2)
        self.fft_A[adapter_name] = conv_layer(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.fft_B[adapter_name] = conv_layer(r, self.out_features, out_kernel, out_stride, bias=False)

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
        if self.fft_magnitude_vector is None:
            # first dora layer being added, add fft_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("fft_magnitude_vector",)

        dora_layer_class = self._get_dora_layer_class()
        dora_layer = dora_layer_class(fan_in_fan_out=False)
        fft_A = self.fft_A[adapter_name].weight
        fft_B = self.fft_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), fft_A=fft_A, fft_B=fft_B, scaling=scaling)
        self.fft_magnitude_vector[adapter_name] = dora_layer

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
            if active_adapter in self.fft_A.keys():
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
                            self.fft_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.fft_magnitude_vector[active_adapter].weight / weight_norm
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
                            self.fft_magnitude_vector[active_adapter]
                            .get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.fft_magnitude_vector[active_adapter].weight / weight_norm
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
            if active_adapter in self.fft_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.fft_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(*self._get_dora_factor_view()) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.fft_B[adapter].weight.device
        dtype = self.fft_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.fft_A[adapter].weight
        weight_B = self.fft_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/fft.py#L117
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
            self.fft_A[adapter].weight.data = weight_A.to(dtype)
            self.fft_B[adapter].weight.data = weight_B.to(dtype)

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
                if active_adapter not in self.fft_A.keys():
                    continue
                fft_A = self.fft_A[active_adapter]
                fft_B = self.fft_B[active_adapter]
                dropout = self.fft_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(fft_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + fft_B(fft_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.fft_magnitude_vector[active_adapter](
                        x,
                        fft_A=fft_A,
                        fft_B=fft_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fft." + rep


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
    fft_config: FFTConfig,
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
        embedding_kwargs.update(fft_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(fft_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        kwargs.update(fft_config.loftq_config)
        new_module = Conv3d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = fft_config.fan_in_fan_out = False
        kwargs.update(fft_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = fft_config.fan_in_fan_out = True
        kwargs.update(fft_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
