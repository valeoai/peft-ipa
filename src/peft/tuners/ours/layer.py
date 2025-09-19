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

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict

import logging


class OursLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("ours_code", )
    other_param_names = ("ours_A", "ours_B", "ours_tensor")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.ours_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.ours_code = nn.ParameterDict({})

        # Stores a reference to the ours_A/B BufferDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.ours_A: Optional[BufferDict] = None
        self.ours_B: Optional[BufferDict] = None
        self.ours_tensor: Optional[BufferDict] = None

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        ours_A: BufferDict,
        ours_B: BufferDict,
        ours_tensor: BufferDict,
        r,
        dim_code,
        ours_dropout,
        init_weights
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if ours_dropout > 0.0:
            ours_dropout_layer = nn.Dropout(p=ours_dropout)
        else:
            ours_dropout_layer = nn.Identity()
 
        self.ours_dropout.update(nn.ModuleDict({adapter_name: ours_dropout_layer}))
        # Actual trainable parameters
        self.ours_code[adapter_name] = nn.Parameter(torch.zeros(dim_code), requires_grad=True)

        # non trainable references to ours_A/B buffers
        self.ours_A = ours_A
        self.ours_B = ours_B
        self.ours_tensor = ours_tensor
        if adapter_name not in ours_A:
            # This means that this is not the first Ours adapter. We have to add an entry in the dict for this adapter.
            if len(self.ours_A) < 1:
                raise ValueError(
                    "The `ours_A` and `ours_B` buffers are empty. This should not happen. Please report this issue."
                )
            # we can take any of the existing adapter's parameters, as they should all be identical
            ours_A_param = list(self.ours_A.values())[0]
            ours_B_param = list(self.ours_B.values())[0]
            ours_tensor_param = list(self.ours_tensor.values())[0]

            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional Ours "
                "adapter was added after the first one with incompatible shapes."
            )
            # check input size
            if ours_A_param.shape[1] < self.in_features:
                raise ValueError(error_tmpl.format("ours_A", ours_A_param.shape[1], self.in_features))
            # check output size
            if ours_B_param.shape[0] < self.out_features:
                raise ValueError(error_tmpl.format("ours_B", ours_B_param.shape[0], self.out_features))
            # check r
            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional Ours "
                "adapter with a lower rank was added after the first one; loading the adapters "
                "in reverse order may solve this."
            )
            if ours_A_param.shape[0] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("ours_A", ours_A_param.shape[0], self.r[adapter_name]))
            if ours_B_param.shape[1] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("ours_B", ours_B_param.shape[1], self.r[adapter_name]))

            self.ours_A[adapter_name] = ours_A_param
            self.ours_B[adapter_name] = ours_B_param
            self.ours_tensor[adapter_name] = ours_tensor_param 

        if init_weights:
            self.reset_ours_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_ours_parameters(self, adapter_name):
        if adapter_name in self.ours_code.keys():
            with torch.no_grad():
                nn.init.zeros_(self.ours_code[adapter_name])


class Linear(nn.Linear, OursLayer):
    # Ours implemented in a dense layer
    def __init__(
        self,
        base_layer,
        ours_A: BufferDict,
        ours_B: BufferDict,
        ours_tensor: BufferDict,
        adapter_name: str,
        r: int = 0,
        dim_code: int = 0,
        ours_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        OursLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, ours_A, ours_B, ours_tensor, r, dim_code, ours_dropout, init_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.ours_lambda_d.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.ours_lambda_d.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        ours_A = self.ours_A[adapter]
        ours_B = self.ours_B[adapter]
        ours_tensor = self.ours_tensor[adapter]

        device = ours_B.device
        dtype = ours_B.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        code = self.ours_code[adapter]

        if cast_to_fp32:
            ours_A = ours_A.float()
            ours_B = ours_B.float()
            code = code.float()

        sliced_A = ours_A[:, : self.in_features]
        sliced_B = ours_B[: self.out_features, :]
        code = code.unsqueeze(-1)

        output_tensor = transpose(sliced_B @ (ours_tensor @ code) @ sliced_A, self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            self.ours_code[adapter].data = code.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ours_code.keys():
                    continue

                code = self.ours_code[active_adapter]

                ours_A = self.ours_A[active_adapter]
                ours_B = self.ours_B[active_adapter]
                ours_tensor = self.ours_tensor[active_adapter]
                # ours_tensor = ours_tensor + torch.diag(torch.ones(32).cuda()).unsqueeze(-1)

                # As adapted layers may have different shapes and Ours contains a single shared pair of A and B matrices,
                # we initialize these matrices with the largest required size for each dimension.
                # During the forward pass, required submatrices are sliced out from the shared ours_A and ours_B.
                sliced_A = ours_A[:, :self.in_features]
                sliced_B = ours_B[:self.out_features, :]

                dropout = self.ours_dropout[active_adapter]
                x = x.to(code.dtype)
                # result = result + lambda_b * F.linear(lambda_d * F.linear(dropout(x), sliced_A), sliced_B)
                conditioned_weight = torch.einsum('ijc,c->ij', ours_tensor, code)
                result = result + F.linear(F.linear(F.linear(dropout(x), sliced_A), conditioned_weight), sliced_B)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ours." + rep
