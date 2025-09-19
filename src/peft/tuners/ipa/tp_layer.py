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

import importlib
import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.init as init

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose
from peft.utils.integrations import gather_params_ctx

from .layer import IPALayer


class IPAParallelLinear(nn.Module, IPALayer):
    """
    When the target layer parallel_linear is RowParallelLinear, in order to keep the input and output shapes
    consistent, we need to split the ipa matrix A into rows, and the ipa_B at this time should be a complete linear
    layer; In the same way, when the target layer is ColumnParallelLinear, we perform column segmentation on ipa_B,
    while ipa_A is still a complete linear layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str, 
        backend,
        r: int = 0,
        ipa_alpha: int = 1,
        ipa_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_ipa_weights: Union[bool, str] = True,
        use_rsipa: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        super().__init__()
        IPALayer.__init__(self, base_layer=base_layer, **kwargs)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        self.backend = backend
        self.is_parallel_a = isinstance(base_layer, backend.RowParallelLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        megatron_config = kwargs["megatron_config"]
        parallel_linear_kwargs = {"megatron_config": megatron_config}
        init_method = init.xavier_normal_
        if hasattr(megatron_config, "init_method"):
            init_method = megatron_config.init_method
        input_is_parallel = True
        gather_output = False
        if isinstance(base_layer, self.backend.RowParallelLinear):
            input_is_parallel = base_layer.input_is_parallel
        else:
            gather_output = base_layer.gather_output
        self.update_layer(
            adapter_name,
            r,
            ipa_alpha=ipa_alpha,
            ipa_dropout=ipa_dropout,
            init_ipa_weights=init_ipa_weights,
            use_rsipa=use_rsipa,
            use_dora=use_dora,
            init_method=init_method,
            input_is_parallel=input_is_parallel,
            gather_output=gather_output,
            **parallel_linear_kwargs,
        )

        if is_target_conv_1d_layer:
            raise ValueError(
                f"{self.__class__.__name__} does not support target_conv_1d_layer yet, please set it to False"
            )
        self.is_target_conv_1d_layer = False

    def update_layer(
        self,
        adapter_name,
        r,
        ipa_alpha,
        ipa_dropout,
        init_ipa_weights,
        use_rsipa,
        use_dora=False,
        init_method=init.xavier_normal_,
        input_is_parallel=True,
        gather_output=False,
        **parallel_linear_kwargs,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.ipa_alpha[adapter_name] = ipa_alpha
        if ipa_dropout > 0.0:
            ipa_dropout_layer = nn.Dropout(p=ipa_dropout)
        else:
            ipa_dropout_layer = nn.Identity()

        self.ipa_dropout[adapter_name] = ipa_dropout_layer

        megatron_config = parallel_linear_kwargs["megatron_config"]
        # ipa needs to be forced to upgrade to 32-bit precision, otherwise it will overflow
        megatron_config.params_dtype = torch.float32
        if self.is_parallel_a:
            ipa_a = self.backend.RowParallelLinear(
                input_size=self.in_features,
                output_size=r,
                bias=False,
                input_is_parallel=input_is_parallel,
                skip_bias_add=True,
                init_method=init_method,
                config=megatron_config,
            )
            ipa_b = nn.Linear(in_features=r, out_features=self.out_features, bias=False, dtype=torch.float32)
        else:
            ipa_a = nn.Linear(in_features=self.in_features, out_features=r, bias=False, dtype=torch.float32)
            ipa_b = self.backend.ColumnParallelLinear(
                input_size=r,
                output_size=self.out_features,
                bias=False,
                gather_output=gather_output,
                init_method=init_method,
                config=megatron_config,
            )
        self.ipa_A[adapter_name] = ipa_a
        self.ipa_B[adapter_name] = ipa_b
        if use_rsipa:
            self.scaling[adapter_name] = ipa_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = ipa_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_ipa_weights, str) and init_ipa_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_ipa_weights)
        elif isinstance(init_ipa_weights, str) and init_ipa_weights.lower() == "oipa":
            with gather_params_ctx(self.get_base_layer().weight):
                self.oipa_init(adapter_name)
        elif init_ipa_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_ipa_weights:
            self.reset_ipa_parameters(adapter_name, init_ipa_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        # If weight is used for matrix multiplication here, the final aggregation operation of the original
        # parallel_linear layer will be missing, so we need to directly call its forward function to obtain the
        # output of the original parallel_linear layer.
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result, bias = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise ValueError(f"{self.__class__.__name__} does not support mixed_batch_forward yet.")
        elif self.merged:
            result, bias = self.base_layer(x, *args, **kwargs)
        else:
            result, bias = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ipa_A.keys():
                    continue
                ipa_A = self.ipa_A[active_adapter]
                ipa_B = self.ipa_B[active_adapter]
                dropout = self.ipa_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(ipa_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    ipa_result = ipa_A(dropout(x))
                    if isinstance(ipa_result, tuple):
                        ipa_result = ipa_result[0]
                    ipa_result = ipa_B(ipa_result)
                    if isinstance(ipa_result, tuple):
                        ipa_result = ipa_result[0]
                    ipa_result = ipa_result * scaling

                    result = result + ipa_result
                else:
                    x = dropout(x)
                    result = result + self.ipa_magnitude_vector[active_adapter](
                        x,
                        ipa_A=ipa_A,
                        ipa_B=ipa_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
        return result, bias

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
            if active_adapter in self.ipa_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.ipa_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.ipa_magnitude_vector[active_adapter].weight / weight_norm
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
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.ipa_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.ipa_magnitude_vector[active_adapter].weight / weight_norm
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
            if active_adapter in self.ipa_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.ipa_magnitude_vector[active_adapter].weight / weight_norm
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

        weight_A = self.ipa_A[adapter].weight
        weight_B = self.ipa_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.ipa_A[adapter].weight.data = weight_A.to(dtype)
            self.ipa_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ipa." + rep


def dispatch_megatron(
    target: torch.nn.Module,
    adapter_name: str,
    ipa_config,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if ipa_config.megatron_config:
        megatron_core = importlib.import_module(ipa_config.megatron_core)
    else:
        megatron_core = None

    if megatron_core and isinstance(
        target_base_layer,
        (megatron_core.tensor_parallel.ColumnParallelLinear, megatron_core.tensor_parallel.RowParallelLinear),
    ):
        megatron_kwargs = kwargs.copy()
        megatron_config = ipa_config.megatron_config
        if isinstance(megatron_config, dict):
            transformer_config_class = megatron_core.transformer.transformer_config.TransformerConfig
            megatron_config = transformer_config_class(**ipa_config.megatron_config)
        megatron_kwargs["megatron_config"] = megatron_config
        if megatron_kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `ColumnParallelLinear` "
                "or `RowParallelLinear`. "
                "Setting fan_in_fan_out to False."
            )
            megatron_kwargs["fan_in_fan_out"] = ipa_config.fan_in_fan_out = False
        new_module = IPAParallelLinear(
            base_layer=target, adapter_name=adapter_name, backend=megatron_core.tensor_parallel, **megatron_kwargs
        )

    return new_module
