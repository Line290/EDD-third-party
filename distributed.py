# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable


def allreduce_params(model, reduce_after=False, no_scale=False, fp32_allreduce=True):
    buckets = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            tp = (param.data.type())
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(param)

    for tp in buckets:
        bucket = buckets[tp]
        grads = [param.grad.data for param in bucket]
        coalesced = _flatten_dense_tensors(grads)
        if fp32_allreduce:
            coalesced = coalesced.float()
        if not no_scale and not reduce_after:
            coalesced /= dist.get_world_size()
        dist.all_reduce(coalesced)
        torch.cuda.synchronize()
        if not no_scale and reduce_after:
            coalesced /= dist.get_world_size()
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)


def allreduce_params_opt(optimizer, reduce_after=False, no_scale=False, fp32_allreduce=False):
    buckets = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.requires_grad and param.grad is not None:
                tp = (param.data.type())
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)

    for tp in buckets:
        bucket = buckets[tp]
        grads = [param.grad.data for param in bucket]
        coalesced = _flatten_dense_tensors(grads)
        if fp32_allreduce:
            coalesced = coalesced.float()
        if not no_scale and not reduce_after:
            coalesced /= dist.get_world_size()
        dist.all_reduce(coalesced)
        torch.cuda.synchronize()
        if not no_scale and reduce_after:
            coalesced /= dist.get_world_size()
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

