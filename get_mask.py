import inspect
import math
import os
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import numpy as np
from tqdm.auto import tqdm
import transformers
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
import re
# from metrics import calculate_metric

import numpy as np
import torch
from torch import nn

@torch.no_grad()
def get_outlier_masks(self, model, percentile=0.005):
    mask_dict = dict()
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
        top_cutoff = int(p.numel() * percentile)
        mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
        mask[-p.view(-1).abs().argsort()[:top_cutoff]] = True
        mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]
    return mask_dict

@torch.no_grad()
def get_smaller_weight_masks(self, model, percentile=0.20):
    mask_dict = dict()
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
        top_cutoff = int(p.numel() * percentile)
        mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
        mask[(p.view(-1).abs()).argsort()[:top_cutoff]] = True
        mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask].clone()
        del mask
        torch.cuda.empty_cache()
    return mask_dict


@torch.no_grad()
def get_random_masks(self, model, percentile=0.005):
    mask_dict = dict()
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
        random_indices = torch.randperm(p.numel(), device=p.device)[:int(p.numel() * percentile)]
        mask_dict[n] = random_indices.clone()
        torch.cuda.empty_cache()
    return mask_dict


@torch.no_grad()
def get_GraSP_mask(self, model: nn.Module,
                        percentile=5e-3, microbatch=1, minibatch=16):
    assert minibatch % microbatch == 0
    count = 0
    old_batch_size = self._train_batch_size
    self._train_batch_size = microbatch

    weights = []
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: 
            p.requires_grad_(False)
        else:
            weights.append(p)

    hessian_grad_prod = {n : torch.zeros_like(p, device='cpu') 
                            for n, p in model.named_parameters() if p.requires_grad}
    
    weight_params = {n : p for n, p in model.named_parameters()}

    with torch.enable_grad():
        for sampled_batch in self.get_train_dataloader():
            sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}    
            count += 1
            outputs = model(**sampled_batch)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = outputs[0]
            loss /= (minibatch/microbatch)
            grads = torch.autograd.grad(loss, weights, 
                                        create_graph=True, retain_graph=True)

            inner_prod = torch.tensor(0.0, requires_grad=True, device=grads[0].device, dtype=torch.bfloat16)
            for g in grads:
                inner_prod = inner_prod + torch.inner(g.view(-1), g.data.view(-1))
            del grads
            
            inner_prod.backward()
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                hessian_grad_prod[n] = (hessian_grad_prod[n].cuda() + p.grad).cpu()
                p.grad = None
            
            torch.cuda.empty_cache()
            
            if count >= minibatch: 
                break
    
    mask_dict = dict()

    with torch.no_grad():
        for n, g in hessian_grad_prod.items():
            per_layer_scores = -g.cuda() * weight_params[n]
            top_cutoff = int(g.numel() * percentile)
            mask = torch.zeros(g.numel(), dtype=torch.bool)
            mask[per_layer_scores.view(-1).argsort()[:top_cutoff]] = True
            mask_dict[n] = torch.arange(g.numel())[mask].to(device='cuda')

    self._train_batch_size = old_batch_size
    return mask_dict


@torch.no_grad()
def get_gradient_masks(model: nn.Module, train_dataloader,
                                percentile=5e-3, microbatch=1, minibatch=4):
    assert minibatch % microbatch == 0
    count = 0
    print("execute get_gradient_masks")
    # old_batch_size = self._train_batch_size
    # self._train_batch_size = microbatch
    with torch.enable_grad():
        for step, sampled_batch in enumerate(train_dataloader):
            # print(f"Memory Allocated2.1: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}   
            # print(f"Memory Allocated2.2: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            count += 1
            outputs = model(**sampled_batch)
            # print(f"Memory Allocated2.3: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = outputs[0]
            loss /= (minibatch/microbatch)
            loss.backward()
            del sampled_batch
            if count >= minibatch: 
                break
    
    mask_dict = dict()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
            per_layer_scores = p.grad ** 2
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
            mask[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
            mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]
            p.grad = None
    torch.cuda.empty_cache()
    # self._train_batch_size = old_batch_size
    return mask_dict

def extract_layer_number_from_key(key):
    match = re.search(r'\blayer\.([0-9]+)\b', key)
    if match:
        return int(match.group(1))
    else:
        return None

@torch.no_grad()
def get_C4_gradient_masks(model, C4_grad_addr, percentile):
    print("execute get_C4_gradient_masks: ", C4_grad_addr,' ', percentile)
    mask_dict = dict()
    grad = transformers.AutoModelForCausalLM.from_pretrained(
        C4_grad_addr,
    )
    grad_dict = {n : p for n, p in grad.named_parameters()}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
            per_layer_scores = grad_dict[n].cuda()
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
            mask[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
            mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]
    selected_grad_dict = apply_mask_to_grad(grad_dict, mask_dict)

    flat_grad = flatten_grad_dict(selected_grad_dict)

    return mask_dict, flat_grad

def flatten_grad_dict(selected_grad_dict):
    """
    将 selected_grad_dict 中的所有梯度向量拼接成一个大的一维张量。

    Args:
        selected_grad_dict (dict): 筛选后的梯度字典，每个键对应一个梯度向量。

    Returns:
        flat_grad (torch.Tensor): 拼接后的大的一维张量。
    """
    # 提取 selected_grad_dict 中的所有张量，并拼接成一维张量
    flat_grad = torch.cat([grad.view(-1) for grad in selected_grad_dict.values()], dim=0)
    return flat_grad

@torch.no_grad()
def apply_mask_to_grad(grad_dict, mask_dict):

    selected_grad_dict = {}
    
    for layer_name, mask_indices in mask_dict.items():
        # 检查该层是否存在于 grad_dict 中
        if layer_name not in grad_dict:
            print(f"Warning: {layer_name} not found in grad_dict")
            continue
        
        # 获取该层对应的梯度
        grad_layer = grad_dict[layer_name]

        mask_indices = mask_indices.to(grad_layer.device)
        
        # 检查 mask_indices 是否与梯度形状兼容
        if grad_layer.numel() < mask_indices.numel():
            print(f"Error: Mask indices for {layer_name} exceed parameter size.")
            continue
        
        # 根据 mask_indices 筛选梯度值
        selected_grad = grad_layer.view(-1)[mask_indices]
        
        # 保存筛选后的梯度
        selected_grad_dict[layer_name] = selected_grad
    
    return selected_grad_dict


# https://arxiv.org/pdf/2002.07376.pdf, GraSP paper
def get_hessian_grad_weight_product_masks(self, model: nn.Module, 
                                percentile=5e-3, microbatch=16, minibatch=1024):
    assert minibatch % microbatch == 0
    count = 0
    old_batch_size = self._train_batch_size
    self._train_batch_size = microbatch
    parameters_as_tuple = tuple(p for n, p in model.named_parameters() 
                                if p.requires_grad and \
                                    ('weight' in n and 'layer_norm' not in n and 'embed' not in n and 'lm_head' not in n and 'norm' not in n))
    grads = tuple(torch.zeros_like(p) for p in parameters_as_tuple)
    with torch.enable_grad():
        for sampled_batch in self.get_train_dataloader():  
            sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}    
            count += len(sampled_batch['input_ids'])
            loss = model(**sampled_batch).loss
            loss /= (minibatch/microbatch)
            new_grads = torch.autograd.grad(loss, parameters_as_tuple)
            for i, g in enumerate(new_grads):
                grads[i].add_(g)
            del new_grads
            if count >= minibatch: 
                break
    
        torch.cuda.empty_cache()
        mask_dict = dict()
        grad_weight_inner_prod = sum(
            (parameters_as_tuple[i] * grads[i]).sum() for i, g in enumerate(grads)
        )
        hessian_grad_prod = torch.autograd.grad(grad_weight_inner_prod, parameters_as_tuple)
        del grads
    
    layer_count = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n: continue
            per_layer_scores = -(hessian_grad_prod[layer_count].float() * p.data.float())
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
            mask[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
            mask_dict[n] = torch.arange(p.numel(), dtype=torch.int64, device=p.device)[mask]
            layer_count += 1

    self._train_batch_size = old_batch_size
    return mask_dict

# @torch.no_grad()
# def get_C4_gradient_masks(model, C4_grad_addr, percentile):
#     print("execute get_C4_gradient_masks: ", C4_grad_addr,' ', percentile)
#     mask_dict = dict()
#     grad = transformers.AutoModelForCausalLM.from_pretrained(
#         C4_grad_addr,
#     )
#     grad_dict = {n : p for n, p in grad.named_parameters()}
#     specific_grad_key = "model.layers.0.self_attn.q_proj.weight"
#     specific_grad_dict = {}
#     if specific_grad_key in grad_dict:
#         specific_grad_dict[specific_grad_key] = grad_dict[specific_grad_key]
#     with torch.no_grad():
#         for n, p in model.named_parameters():
#             if not p.requires_grad: continue
#             if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
#             per_layer_scores = grad_dict[n].cuda() ** 2
#             top_cutoff = int(p.numel() * percentile)
#             mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
#             mask[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
#             mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]

#     filtered_specific_grad_dict = {}

#     for key in specific_grad_dict:
#         if key in mask_dict:
#             indices = mask_dict[key]
#             # 确保张量在同一设备上
#             values = specific_grad_dict[key].to(indices.device)
#             # 展平成一维
#             flattened_values = values.view(-1)
#             # 筛选
#             filtered_values = flattened_values[indices]
#             # 如果需要，重塑回原始形状
#             # filtered_values = filtered_values.view(some_shape)
#             filtered_specific_grad_dict[key] = filtered_values
#             torch.set_printoptions(threshold=float('inf'))
#             print("filtered_values: ", filtered_values)
#         else:
#             print(f"Key {key} not found in mask_dict.")

#     return mask_dict

# @torch.no_grad()
# def get_C4_gradient_masks(model, C4_grad_addr, percentile):
#     print("execute get_C4_gradient_masks: ", C4_grad_addr,' ', percentile)
#     mask_dict = dict()
#     grad = transformers.AutoModelForCausalLM.from_pretrained(
#         C4_grad_addr,
#     )
#     grad_dict = {n : p for n, p in grad.named_parameters()}
#     with torch.no_grad():
#         for n, p in model.named_parameters():
#             if not p.requires_grad: continue
#             if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
#             per_layer_scores = grad_dict[n].cuda() ** 2
#             top_cutoff = int(p.numel() * percentile)
#             mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
#             mask[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
#             mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]

#     c4_pretrain_grad = {}
#     specific_grad_key = "model.layers.0.self_attn.q_proj.weight"

#     # 检查特定的梯度键是否在 grad_dict 和 mask_dict 中
#     if specific_grad_key in grad_dict and specific_grad_key in mask_dict:
#         # 获取特定层的梯度
#         specific_grad = grad_dict[specific_grad_key].cuda()
#         # 获取对应的掩码索引
#         mask_indices = mask_dict[specific_grad_key]
#         # 将梯度展开为一维，然后应用掩码
#         masked_grad = specific_grad.view(-1)[mask_indices]
#         # 如果需要，还可以将 masked_grad 重新 reshape 回原来的形状
#         c4_pretrain_grad[specific_grad_key] = masked_grad
#     else:
#         print(f"键 {specific_grad_key} 不存在于 grad_dict 或 mask_dict 中")

#     return mask_dict, c4_pretrain_grad