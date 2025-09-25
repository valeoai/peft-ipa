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

import sklearn.decomposition
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils.imports import is_xpu_available
from transformers.pytorch_utils import Conv1D
import sklearn
import numpy as np
import os.path as op

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import IPAConfig
from .._buffer_dict import BufferDict
from .dora import DoraLinearLayer

from typing import Optional, Tuple

class IncrementalPCA:
    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: Optional[bool] = True,
        batch_size: Optional[int] = None,
        svd_driver: Optional[str] = None,
        lowrank: bool = False,
        lowrank_q: Optional[int] = None,
        lowrank_niter: int = 4,
        lowrank_seed: Optional[int] = None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.batch_size = batch_size
        self.svd_driver = svd_driver
        self.lowrank = lowrank
        self.lowrank_q = lowrank_q
        self.lowrank_niter = lowrank_niter
        self.lowrank_seed = lowrank_seed

        self.n_features_ = None

        if self.lowrank:
            self._validate_lowrank_params()

    def _validate_lowrank_params(self):
        if self.lowrank_q is None:
            if self.n_components is None:
                raise ValueError("n_components must be specified when using lowrank mode with lowrank_q=None.")
            self.lowrank_q = self.n_components * 2
        elif self.lowrank_q < self.n_components:
            raise ValueError("lowrank_q must be greater than or equal to n_components.")

    def _svd_fn_full(self, X):
        return torch.linalg.svd(X, full_matrices=False, driver=self.svd_driver)

    def _svd_fn_lowrank(self, X):
        seed_enabled = self.lowrank_seed is not None
        with torch.random.fork_rng(enabled=seed_enabled):
            if seed_enabled:
                torch.manual_seed(self.lowrank_seed)
            U, S, V = torch.svd_lowrank(X, q=self.lowrank_q, niter=self.lowrank_niter)
            return U, S, V.mH

    def _validate_data(self, X) -> torch.Tensor:
        """
        Validates and converts the input data `X` to the appropriate tensor format.

        Args:
            X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Converted to appropriate format.
        """
        valid_dtypes = [torch.float32, torch.float64]

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        elif self.copy:
            X = X.clone()

        n_samples, n_features = X.shape
        if self.n_components is None:
            pass
        elif self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} invalid for n_features={n_features}, "
                "need more rows than columns for IncrementalPCA processing."
            )
        elif self.n_components > n_samples:
            raise ValueError(
                f"n_components={self.n_components} must be less or equal to the batch number of samples {n_samples}"
            )

        if X.dtype not in valid_dtypes:
            X = X.to(torch.float32)

        return X

    @staticmethod
    def _incremental_mean_and_var(
        X, last_mean, last_variance, last_sample_count
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the incremental mean and variance for the data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            last_mean (torch.Tensor): The previous mean tensor with shape (n_features,).
            last_variance (torch.Tensor): The previous variance tensor with shape (n_features,).
            last_sample_count (torch.Tensor): The count tensor of samples processed before the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated mean, variance tensors, and total sample count.
        """
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        if last_sample_count > 0:
            if last_mean is None:
                raise ValueError("last_mean should not be None if last_sample_count > 0.")
            if last_variance is None:
                raise ValueError("last_variance should not be None if last_sample_count > 0.")

        new_sample_count = torch.tensor([X.shape[0]], device=X.device)
        updated_sample_count = last_sample_count + new_sample_count

        if last_mean is None:
            last_sum = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
        else:
            last_sum = last_mean * last_sample_count

        new_sum = X.sum(dim=0, dtype=torch.float64)

        updated_mean = (last_sum + new_sum) / updated_sample_count

        T = new_sum / new_sample_count
        temp = X - T
        correction = temp.sum(dim=0, dtype=torch.float64).square()
        temp.square_()
        new_unnormalized_variance = temp.sum(dim=0, dtype=torch.float64)
        new_unnormalized_variance -= correction / new_sample_count
        if last_variance is None:
            updated_variance = new_unnormalized_variance / updated_sample_count
        else:
            last_unnormalized_variance = last_variance * last_sample_count
            last_over_new_count = last_sample_count.double() / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count / updated_sample_count * (last_sum / last_over_new_count - new_sum).square()
            )
            updated_variance = updated_unnormalized_variance / updated_sample_count

        return updated_mean, updated_variance, updated_sample_count

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for deterministic output.

        This method ensures that the output remains consistent across different runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular vectors to determine the sign flipping.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs[: u.shape[1]].view(1, -1)
        v *= signs.view(-1, 1)
        return u, v

    def fit(self, X, check_input=True):
        """
        Fits the model with data `X` using minibatches of size `batch_size`.

        Args:
            X (torch.Tensor): The input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to True.

        Returns:
            IncrementalPCA: The fitted IPCA model.
        """
        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size = 5 * n_features

        for batch in self.gen_batches(n_samples, self.batch_size, min_batch_size=self.n_components or 0):
            self.partial_fit(X[batch], check_input=False)

        return self

    def partial_fit(self, X, check_input=True):
        """
        Incrementally fits the model with batch data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to True.

        Returns:
            IncrementalPCA: The updated IPCA model after processing the batch.
        """
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        # Initialize attributes to avoid errors during the first call to partial_fit
        if first_pass:
            self.mean_ = None  # Will be initialized properly in _incremental_mean_and_var based on data dimensions
            self.var_ = None  # Will be initialized properly in _incremental_mean_and_var based on data dimensions
            self.n_samples_seen_ = torch.tensor([0], device=X.device)
            self.n_features_ = n_features
            if not self.n_components:
                self.n_components = min(n_samples, n_features)

        if n_features != self.n_features_:
            raise ValueError(
                "Number of features of the new batch does not match the number of features of the first batch."
            )

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, self.n_samples_seen_
        )

        print(col_mean.shape, col_var.shape, n_total_samples)
        print(col_mean.dtype, col_var.dtype, col_var.dtype)

        if first_pass:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt((self.n_samples_seen_.double() / n_total_samples) * n_samples)
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)
            X = torch.vstack(
                (
                    self.singular_values_.view((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )

        if self.lowrank:
            U, S, Vt = self._svd_fn_lowrank(X)
        else:
            U, S, Vt = self._svd_fn_full(X)
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)
        print(U.dtype, Vt.dtype)
        explained_variance = S**2 / (n_total_samples - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[: self.n_components]
        self.singular_values_ = S[: self.n_components]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components]
        if self.n_components not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components :].mean()
        else:
            self.noise_variance_ = torch.tensor(0.0, device=X.device)
        return self

    def transform(self, X) -> torch.Tensor:
        """
        Applies dimensionality reduction to `X`.

        The input data `X` is projected on the first principal components previously extracted from a training set.

        Args:
            X (torch.Tensor): New data tensor with shape (n_samples, n_features) to be transformed.

        Returns:
            torch.Tensor: Transformed data tensor with shape (n_samples, n_components).
        """
        X = X - self.mean_
        return torch.mm(X.double(), self.components_.T).to(X.dtype)

class LinearProj(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    basis: torch.Tensor
    running_mean: torch.Tensor
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
        self.pca = IncrementalPCA(n_components=out_features, lowrank=True, lowrank_niter=3, lowrank_q=out_features, lowrank_seed=42)
        self.fixed_basis = fixed_basis
        self.n_samples_seen = torch.tensor([0], **factory_kwargs)
        self.basis = nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs), requires_grad=not fixed_basis)
        self.register_buffer("running_mean", torch.zeros((1, in_features), **factory_kwargs))
        self.register_buffer("is_first_run", torch.tensor(True, **factory_kwargs))
        self.register_buffer("P", torch.eye(out_features, **factory_kwargs))
        self.register_buffer("normalized_proj", None)

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.basis)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.fixed_basis:
            with torch.no_grad():
                return F.linear(input, self.basis)
        else:
            return F.linear(input, self.basis)
    
    @torch.no_grad()
    def batched_incremental_pca_update(self, x: torch.Tensor):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            dim = x.size(-1)
            reshaped_x = x.reshape(-1, dim)
            self.pca.partial_fit(reshaped_x)
            self.basis.data = self.pca.components_.contiguous().to(self.basis.data.dtype)

    @torch.no_grad()
    def batched_hebbian_update(self, x: torch.Tensor, lr=1e-3):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x = x.reshape(-1, x.size(-1))
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_size = x.size(0)

            # initialize counters on first run
            if self.is_first_run:
                # total_count: how many samples seen so far
                self.total_count = torch.tensor(0, device=x.device)
                self.running_mean = torch.zeros_like(batch_mean)
                self.is_first_run.fill_(False)

            # compute the new total count
            new_count = self.total_count + batch_size
            self.running_mean += (batch_mean - self.running_mean) * (batch_size / new_count)

            self.total_count = new_count

            x = x - self.running_mean
            y = x @ self.basis.data.t()

            batch_size = x.size(0)
            YY = (y.t() @ y) / batch_size
            delta = (y.t() @ x) / batch_size - torch.tril(YY) @ self.basis.data
            
            self.basis.data = self.basis.data + lr * delta
            Q, _ = torch.linalg.qr(self.basis.t())
            self.basis.data = Q.t().contiguous()

    @torch.no_grad()
    def batched_past(self, x: torch.Tensor, lbd=1e-3, beta=0.999):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            size_in = x.size(-1)
            x = x.reshape(-1, size_in)
            batch_mean = x.mean(dim=0, keepdim=True)
            if self.is_first_run:
                self.running_mean = batch_mean
                self.is_first_run.fill_(False)
            else:
                self.running_mean = (1.0 - lbd) * self.running_mean + lbd * batch_mean
            x = x - self.running_mean
            batch_size = x.size(0)
            
            y = F.linear(x, self.basis.data)
            h = F.linear(y, self.P)
            m = h / (beta + torch.mean((y * h).sum(dim=-1)))
            upper = torch.triu(self.P - m.t() @ h / batch_size)
            self.P = 1.0 / beta * (upper + upper.T - torch.diag(upper.diag()))
            e = x - F.linear(y, self.basis.data.T)
            self.basis.data = self.basis.data + m.t() @ e / batch_size

            new_basis, _ = torch.linalg.qr(self.basis.data.t())
            self.basis.data = new_basis.t().contiguous()

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
            # n_sample = reshaped_x.size(0)
            # reshaped_x = reshaped_x.tile((self.out_features // n_sample + 1, 1))

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
    adapter_layer_names = ("ipa_A", "ipa_B", "ipa_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "scaling")

    def __init__(self, base_layer: nn.Module, ipa_mode: str, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.scaling = {}
        self.ipa_mode = ipa_mode
        self.ipa_A = nn.ModuleDict({})
        self.ipa_B = nn.ModuleDict({})
        self.ipa_dropout = nn.ModuleDict({})
        # For Embedding layer
        self.ipa_embedding_A = nn.ParameterDict({})
        self.ipa_embedding_B = nn.ParameterDict({})
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
        self.ipa_A[adapter_name] = LinearProj(self.in_features, r, fixed_basis=("requires_grad" not in self.ipa_mode))
        if "requires_grad" not in self.ipa_mode:
            self.adapter_layer_names = ("ipa_B", "ipa_embedding_B")
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
        lora_A = self.ipa_A[adapter_name].basis
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
            self.ipa_A[adapter_name].reset_parameters()
            nn.init.zeros_(self.ipa_B[adapter_name].weight)
        if adapter_name in self.ipa_embedding_B.keys():
            nn.init.normal_(self.ipa_embedding_B[adapter_name])

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
            if active_adapter not in self.ipa_A.keys():
                continue

            ipa_A = self.ipa_A[active_adapter]
            ipa_B = self.ipa_B[active_adapter]
            dropout = self.ipa_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            sub_batch = x[sub_batch_indices_list[i]].to(ipa_B.weight.dtype)
            ipa_output = ipa_B(ipa_A(dropout(sub_batch))) * scaling
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
            if active_adapter in self.ipa_A.keys():
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
            if active_adapter in self.ipa_A.keys():
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

        weight_A = self.ipa_A[adapter].basis
        weight_B = self.ipa_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.ipa_A[adapter].basis.data = weight_A.to(dtype)
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
                    self.ipa_A[active_adapter].batched_incremental_pca_update(x)
                elif self.ipa_mode.startswith("pre_hebbian"):
                    self.ipa_A[active_adapter].batched_hebbian_update(x)
                elif self.ipa_mode.startswith("pre_past"):
                    self.ipa_A[active_adapter].batched_past(x)

            result = self.base_layer(x, *args, **kwargs)

        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ipa_A.keys():
                    continue
                ipa_A = self.ipa_A[active_adapter]
                ipa_B = self.ipa_B[active_adapter]
                ipa_dropout = self.ipa_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(ipa_B.weight.dtype)

                if self.training and self.ipa_mode.startswith("online_hebbian"):
                    ipa_A.batched_hebbian_update(x)
                elif self.training and self.ipa_mode.startswith("online_ipca"):
                    ipa_A.batched_incremental_pca_update(x)
                elif self.training and self.ipa_mode.startswith("online_past"):
                    ipa_A.batched_past(x)
                elif self.training and self.ipa_mode.startswith("rand_ortho"):
                    if ipa_A.is_first_run:
                        with torch.no_grad():
                            nn.init.orthogonal_(ipa_A.basis)
                            ipa_A.is_first_run.fill_(False)
                elif self.training and self.ipa_mode == "moving_avg":
                    ipa_A.batch_basis_moving_average(x)

                if self.ipa_mode.endswith("hira"):
                    if ipa_A.fixed_basis:
                        basis = ipa_A.basis.detach()
                    else:
                        basis = ipa_A.basis
                    result = result + F.linear(ipa_dropout(x), self.base_layer.weight * (ipa_B.weight @ basis).to(self.base_layer.weight.dtype) * scaling)
                else:
                    if not self.use_dora[active_adapter]:
                        result = result + ipa_B(ipa_A(ipa_dropout(x)) * scaling) 
                    else:
                        x = ipa_dropout(x)
                        result = result + self.lora_magnitude_vector[active_adapter](
                            x,
                            lora_A=ipa_A,
                            lora_B=ipa_B,
                            scaling=scaling,
                            base_layer=self.get_base_layer(),
                        )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ipa." + rep


class Embedding(nn.Module, IPALayer):
    # IPA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        ipa_alpha: int = 1,
        ipa_dropout: float = 0.0,
        init_ipa_weights: Union[bool, str] = True,
        use_rsipa: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        IPALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            ipa_alpha=ipa_alpha,
            ipa_dropout=ipa_dropout,
            init_ipa_weights=init_ipa_weights,
            use_rsipa=use_rsipa,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, ipa_alpha, ipa_dropout, init_ipa_weights, use_rsipa, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.ipa_alpha[adapter_name] = ipa_alpha
        if ipa_dropout > 0.0:
            ipa_dropout_layer = nn.Dropout(p=ipa_dropout)
        else:
            ipa_dropout_layer = nn.Identity()

        self.ipa_dropout[adapter_name] = ipa_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.ipa_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.ipa_embedding_B[adapter_name] = nn.Parameter(weight_B)
        if use_rsipa:
            self.scaling[adapter_name] = ipa_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = ipa_alpha / r

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
        if self.ipa_magnitude_vector is None:
            # first dora layer being added, add ipa_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("ipa_magnitude_vector",)

        dora_layer = DoraEmbeddingLayer(fan_in_fan_out=True)
        ipa_embedding_A = self.ipa_embedding_A[adapter_name]
        ipa_embedding_B = self.ipa_embedding_B[adapter_name]
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(), ipa_A=ipa_embedding_A, ipa_B=ipa_embedding_B, scaling=scaling
        )
        self.ipa_magnitude_vector[adapter_name] = dora_layer

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
            if active_adapter in self.ipa_embedding_A.keys():
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
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.ipa_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.ipa_embedding_B[adapter].device
        dtype = self.ipa_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.ipa_embedding_A[adapter]
        weight_B = self.ipa_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.ipa_embedding_A[adapter] = weight_A.to(dtype)
            self.ipa_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.ipa_embedding_A.keys():
                continue

            embedding_A = self.ipa_embedding_A[active_adapter].T
            embedding_B = self.ipa_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
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
                if active_adapter not in self.ipa_embedding_A:
                    continue
                embedding_A = self.ipa_embedding_A[active_adapter].T
                embedding_B = self.ipa_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]

                if not self.use_dora[active_adapter]:
                    after_A = self._embed(x, embedding_A)
                    result = result + (after_A @ embedding_B) * scaling
                else:
                    mag_norm_scale, dora_result = self.ipa_magnitude_vector[active_adapter](
                        x,
                        ipa_A=embedding_A,
                        ipa_B=embedding_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        embed_fn=self._embed,
                    )
                    result = mag_norm_scale * result + dora_result
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ipa." + rep


class _ConvNd(nn.Module, IPALayer):
    # IPA implemented in a conv(2,3)d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        ipa_alpha: int = 1,
        ipa_dropout: float = 0.0,
        init_ipa_weights: Union[bool, str] = True,
        use_rsipa: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        IPALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()

        self.update_layer(
            adapter_name,
            r,
            ipa_alpha=ipa_alpha,
            ipa_dropout=ipa_dropout,
            init_ipa_weights=init_ipa_weights,
            use_rsipa=use_rsipa,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, ipa_alpha, ipa_dropout, init_ipa_weights, use_rsipa, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.ipa_alpha[adapter_name] = ipa_alpha
        if ipa_dropout > 0.0:
            ipa_dropout_layer = nn.Dropout(p=ipa_dropout)
        else:
            ipa_dropout_layer = nn.Identity()

        self.ipa_dropout[adapter_name] = ipa_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        conv_layer = type(base_layer)
        out_kernel = out_stride = (1,) * (self._kernel_dim - 2)
        self.ipa_A[adapter_name] = conv_layer(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.ipa_B[adapter_name] = conv_layer(r, self.out_features, out_kernel, out_stride, bias=False)
        if use_rsipa:
            self.scaling[adapter_name] = ipa_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = ipa_alpha / r


        self.reset_ipa_parameters(adapter_name, init_ipa_weights)

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
        if self.ipa_magnitude_vector is None:
            # first dora layer being added, add ipa_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("ipa_magnitude_vector",)

        dora_layer_class = self._get_dora_layer_class()
        dora_layer = dora_layer_class(fan_in_fan_out=False)
        ipa_A = self.ipa_A[adapter_name].weight
        ipa_B = self.ipa_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), ipa_A=ipa_A, ipa_B=ipa_B, scaling=scaling)
        self.ipa_magnitude_vector[adapter_name] = dora_layer

    def _get_dora_layer_class(self) -> type[_DoraConvNdLayer]:
        # Subclasses should override this method to return the appropriate DoraLayer class
        raise NotImplementedError

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
            if active_adapter in self.ipa_A.keys():
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
                            self.ipa_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.ipa_magnitude_vector[active_adapter].weight / weight_norm
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
                            self.ipa_magnitude_vector[active_adapter]
                            .get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.ipa_magnitude_vector[active_adapter].weight / weight_norm
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
            if active_adapter in self.ipa_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.ipa_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(*self._get_dora_factor_view()) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.ipa_B[adapter].weight.device
        dtype = self.ipa_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.ipa_A[adapter].weight
        weight_B = self.ipa_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/ipa.py#L117
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
            self.ipa_A[adapter].weight.data = weight_A.to(dtype)
            self.ipa_B[adapter].weight.data = weight_B.to(dtype)

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
                if active_adapter not in self.ipa_A.keys():
                    continue
                ipa_A = self.ipa_A[active_adapter]
                ipa_B = self.ipa_B[active_adapter]
                dropout = self.ipa_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(ipa_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + ipa_B(ipa_A(dropout(x))) * scaling
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
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ipa." + rep


class Conv2d(_ConvNd):
    # IPA implemented in a conv2d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 4:
            raise ValueError(f"Conv2d layer kernel must have 4 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv2d

    def _get_dora_layer_class(self):
        return DoraConv2dLayer


class Conv3d(_ConvNd):
    # IPA implemented in a conv3d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 5:
            raise ValueError(f"Conv3d layer kernel must have 5 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv3d

    def _get_dora_layer_class(self):
        return DoraConv3dLayer


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

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        new_module = Conv3d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = ipa_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = ipa_config.fan_in_fan_out = True
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
