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
from typing import Any, Optional, Union, Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils.imports import is_xpu_available
from transformers.pytorch_utils import Conv1D
import numpy as np
import os.path as op

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import IPAConfig
from .._buffer_dict import BufferDict
from .dora import DoraLinearLayer

from typing import Optional, Tuple

def svd_flip(u, v, u_based_decision=True):
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
    else:
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
    u *= signs[: u.shape[1]].view(1, -1)
    v *= signs.view(-1, 1)
    return u, v

def incremental_mean(X, last_mean, last_sample_count):
    new_sample_count = torch.tensor([X.shape[0]], device=X.device)
    updated_sample_count = last_sample_count + new_sample_count
    last_sum = last_mean * last_sample_count if last_sample_count > 0 else torch.zeros_like(last_mean)
    new_sum = X.sum(dim=0, dtype=torch.float64)
    updated_mean = (last_sum + new_sum) / updated_sample_count
    return updated_mean, updated_sample_count

class LinearProj(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    basis: torch.Tensor
    running_mean: torch.Tensor
    running_var: torch.Tensor
    is_first_run: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fixed_basis: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_basis = fixed_basis
        self.register_buffer("n_samples_seen", torch.tensor([0], device=device))
        self.basis = nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs), requires_grad=not fixed_basis)
        self.register_buffer("components", torch.zeros((out_features, in_features), device=device, dtype=torch.float64))
        self.register_buffer("running_mean", torch.zeros(in_features, device=device, dtype=torch.float64))
        self.register_buffer("is_first_run", torch.tensor(True, device=device, dtype=torch.bool))
        self.register_buffer("singular_values", torch.zeros(out_features, device=device, dtype=torch.float64))
        self.register_buffer("normalized_proj", None)

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.basis)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.basis)
    
    @torch.no_grad()
    def batched_ipca_update(self, x: torch.Tensor, svd_niter=3):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            dim = x.size(-1)
            reshaped_x = x.reshape(-1, dim)
            n_samples = reshaped_x.size(0)

            reshaped_x = reshaped_x.clone().to(torch.float32)
 
            col_mean, n_total_samples = incremental_mean(reshaped_x, self.running_mean, self.n_samples_seen)

            if self.is_first_run:
                reshaped_x -= col_mean
                self.is_first_run.fill_(False)
            else:
                col_batch_mean = torch.mean(reshaped_x, dim=0)
                reshaped_x -= col_batch_mean
                mean_correction_factor = torch.sqrt((self.n_samples_seen.double() / n_total_samples) * n_samples)
                mean_correction = mean_correction_factor * (self.running_mean - col_batch_mean)
                reshaped_x = torch.vstack(
                    (
                        self.singular_values.view((-1, 1)) * self.components,
                        reshaped_x,
                        mean_correction,
                    )
                )

            U, S, V = torch.svd_lowrank(reshaped_x, q=self.out_features, niter=svd_niter)
            U, Vt = svd_flip(U, V.mH, u_based_decision=False)
 
            self.n_samples_seen = n_total_samples
            self.singular_values = S[:self.out_features]
            self.components = Vt[:self.out_features].contiguous()
            self.running_mean = col_mean
            self.basis.data = self.components.to(dtype=self.basis.dtype)

    @torch.no_grad()
    def batched_gha_update(self, x: torch.Tensor, lr=1e-3, reortho=False):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            reshaped_x = x.reshape(-1, x.size(-1))
            batch_size = reshaped_x.size(0)

            col_mean, n_total_samples = incremental_mean(reshaped_x, self.running_mean, self.n_samples_seen)
            self.running_mean.copy_(col_mean)
            self.n_samples_seen.copy_(n_total_samples)
            
            x = x - self.running_mean
            y = x @ self.basis.t()

            batch_size = x.size(0)
            YY = (y.t() @ y) / batch_size
            delta = (y.t() @ x) / batch_size - torch.tril(YY) @ self.basis
            
            self.basis += lr * delta
            if reortho:
                Q, _ = torch.linalg.qr(self.basis.t())
                self.basis.copy_(Q.t())

    @torch.no_grad()
    def interpolate_basis(self, incoming_basis, epsilon, reortho=True):
        new_basis_interp = self.basis.data * (1. - epsilon) + incoming_basis * epsilon
        if reortho:
            new_basis, _ = torch.linalg.qr(new_basis_interp.t())
            self.basis.data = new_basis.t().contiguous()
        else:
            self.basis.data = new_basis_interp.contiguous()

    @torch.no_grad()
    def batch_basis_moving_average(self, x: torch.Tensor, epsilon=1e-2):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            dim = x.size(-1)
            reshaped_x = x.reshape(-1, dim)

            _, _, V = torch.pca_lowrank(reshaped_x, q=self.out_features, center=False)
            Vh = V.t().contiguous()

            if self.is_first_run:
                self.basis.data = Vh
                self.is_first_run.fill_(False)
            else:
                self.interpolate_basis(Vh, epsilon)
            
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"

class IPALayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("ipa_P", "ipa_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "scaling")

    def __init__(self, base_layer: nn.Module, ipa_mode: str, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.scaling = {}
        self.ipa_mode = ipa_mode
        self.ipa_P = nn.ModuleDict({})
        self.ipa_B = nn.ModuleDict({})
        self.ipa_dropout = nn.ModuleDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
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
        self, adapter_name, r, scaling, ipa_dropout, use_dora
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.scaling[adapter_name] = scaling
        if ipa_dropout > 0.0:
            ipa_dropout_layer = nn.Dropout(p=ipa_dropout)
        else:
            ipa_dropout_layer = nn.Identity()

        self.ipa_dropout.update(nn.ModuleDict({adapter_name: ipa_dropout_layer}))

        # Actual trainable parameters
        self.ipa_P[adapter_name] = LinearProj(self.in_features, r, fixed_basis=("requires_grad" not in self.ipa_mode))
        if "requires_grad" not in self.ipa_mode:
            self.adapter_layer_names = ("ipa_B", )
        self.ipa_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.scaling[adapter_name] = scaling

        self.reset_ipa_parameters(adapter_name)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)
        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: str) -> None:
        if not self.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, "fan_in_fan_out", False))
        lora_A = self.ipa_P[adapter_name].basis
        lora_B = self.ipa_B[adapter_name].weight
        place_on_cpu = self.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if self.ephemeral_gpu_offload:
            if lora_A.device.type in ["cuda", "xpu"]:
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type not in ["cuda", "xpu"]:
                    if is_xpu_available():
                        lora_B = lora_B.to("xpu")
                    else:
                        lora_B = lora_B.to("cuda")
                lora_A = lora_A.to(lora_B.device)
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling, place_on_cpu=place_on_cpu
        )
        self.lora_magnitude_vector[adapter_name] = dora_layer


    def reset_ipa_parameters(self, adapter_name):
        if adapter_name in self.ipa_B.keys():
            self.ipa_P[adapter_name].reset_parameters()
            nn.init.zeros_(self.ipa_B[adapter_name].weight)

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

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.ipa_P.keys():
                continue

            ipa_P = self.ipa_P[active_adapter]
            ipa_B = self.ipa_B[active_adapter]
            dropout = self.ipa_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            sub_batch = x[sub_batch_indices_list[i]].to(ipa_B.weight.dtype)
            ipa_output = ipa_B(ipa_P(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += ipa_output.to(torch_result_dtype)

        return result


# Below code is based on https://github.com/microsoft/LoRA/blob/main/ipalib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, IPALayer):
    # IPA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        ipa_dropout: float = 0,
        scaling: float = 0.25,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        IPALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            scaling=scaling,
            ipa_dropout=ipa_dropout,
            use_dora=use_dora,
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
            if active_adapter in self.ipa_P.keys():
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
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)
                    
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
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
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
            if active_adapter in self.ipa_P.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.ipa_B[adapter].weight.device
        dtype = self.ipa_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.ipa_P[adapter].basis
        weight_B = self.ipa_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.ipa_P[adapter].basis.data = weight_A.to(dtype)
            self.ipa_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()

            for active_adapter in self.active_adapters:
                if self.ipa_mode.startswith("pre_ipca"):
                    self.ipa_P[active_adapter].batched_ipca_update(x)
                elif self.ipa_mode.startswith("pre_gha"):
                    self.ipa_P[active_adapter].batched_gha_update(x)

            result = self.base_layer(x, *args, **kwargs)

        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ipa_P.keys():
                    continue
                ipa_P = self.ipa_P[active_adapter]
                ipa_B = self.ipa_B[active_adapter]
                ipa_dropout = self.ipa_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(ipa_B.weight.dtype)

                if self.training and self.ipa_mode.startswith("online_gha"):
                    ipa_P.batched_gha_update(x)
                elif self.training and self.ipa_mode.startswith("online_ipca"):
                    ipa_P.batched_ipca_update(x)
                elif self.training and self.ipa_mode.startswith("rand_ortho"):
                    if ipa_P.is_first_run:
                        with torch.no_grad():
                            nn.init.orthogonal_(ipa_P.basis)
                            dim_in = ipa_P.basis.shape[1]
                            dim_out = ipa_P.basis.shape[0]
                            ipa_P.basis.data.mul_(math.sqrt(dim_in / dim_out))
                            ipa_P.is_first_run.fill_(False)
                elif self.training and self.ipa_mode == "moving_avg":
                    ipa_P.batch_basis_moving_average(x)

                if not self.use_dora[active_adapter]:
                    result = result + ipa_B(ipa_P(ipa_dropout(x)) * scaling) 
                else:
                    x = ipa_dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=ipa_P,
                        lora_B=ipa_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ipa." + rep




def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    ipa_config: IPAConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = ipa_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module
