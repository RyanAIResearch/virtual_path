#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
from itertools import chain
import sys

import datasets
import numpy as np
import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from eventProfiler import EventProfiler
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)


logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@torch.no_grad()
def zo_perturb_parameters(args, random_seed=None, scaling_factor=1):
    # Set the random seed to ensure that we sample the same z for perturbation/update
    torch.manual_seed(random_seed if random_seed is not None else args.zo_random_seed)
    for name, param in args.named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        param.add_(z, alpha=(scaling_factor * args.zo_eps))


@torch.no_grad()
def squeezellm_zo_perturb_parameters(args, random_seed=None, scaling_factor=1):
    torch.manual_seed(random_seed if random_seed is not None else args.zo_random_seed)
    
    for name, buffer in args.dequantized_weight_list:
        indices = args.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
        z = torch.normal(mean=0, std=1, size=(indices.shape[1],), device=buffer.device, dtype=buffer.dtype)
        buffer[indices[0], indices[1]] += (scaling_factor * args.zo_eps) * z


@torch.no_grad()
def zo_perturb_parameters_with_mask(args, random_seed=None, mask_dict=None, scaling_factor=1):
    torch.manual_seed(random_seed if random_seed is not None else args.zo_random_seed)
    
    for name, param in args.model.named_parameters():
        if not name in args.named_parameters_to_optim: continue
        selected_param = args.named_parameters_to_optim[name]
        z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
        param.view(-1)[mask_dict[name]] += (scaling_factor * args.zo_eps) * z


@torch.inference_mode()
def set_squeezellm_sparse_grad(args, model, inputs):
    args.zo_random_seed = np.random.randint(100000000000)

    squeezellm_zo_perturb_parameters(args, args.zo_random_seed, scaling_factor=1)
    loss1 = model(**inputs).loss

    squeezellm_zo_perturb_parameters(args, args.zo_random_seed, scaling_factor=-2)
    loss2 = model(**inputs).loss

    args.projected_grad = ((loss1 - loss2) / (2 * args.zo_eps)).item()

    squeezellm_zo_perturb_parameters(args, args.zo_random_seed, scaling_factor=1)
    
    torch.manual_seed(args.zo_random_seed)

    for name, buffer in args.dequantized_weight_list:
        sensitive_vals = args.sqllm_sparse_weight_dict[name]
        z = torch.normal(mean=0, std=1, size=sensitive_vals.size(), device=sensitive_vals.device, dtype=sensitive_vals.dtype)
        sensitive_vals.grad = args.projected_grad * z

    return (loss1 + loss2) / 2


@torch.no_grad()
def get_outlier_masks(model, percentile=0.005):
    mask_dict = dict()
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
        top_cutoff = int(p.numel() * percentile)
        mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
        mask[(-p.abs()).argsort()[:top_cutoff]] = True
        mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]
    return mask_dict


@torch.no_grad()
def get_random_masks(model, percentile=0.005):
    mask_dict = dict()
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
        random_indices = torch.randperm(p.numel(), device=p.device)[:int(p.numel() * percentile)]
        mask_dict[n] = random_indices.clone()
        torch.cuda.empty_cache()
    return mask_dict


def get_gradient_masks(model: torch.nn.Module, train_dataloader,
                        percentile=5e-3, microbatch=1, minibatch=16):
    assert minibatch % microbatch == 0
    count = 0
    for sampled_batch in train_dataloader:
        sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}    
        count += 1
        outputs = model(**sampled_batch)
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        else:
            loss = outputs[0]
        loss /= (minibatch/microbatch)
        loss.backward()
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

    return mask_dict


def zo_step(args, model, inputs):
    """
    Estimate gradient by MeZO. 
    """
    args.named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            args.named_parameters_to_optim.append((name, param))

    # Sample the random seed for sampling z
    args.zo_random_seed = np.random.randint(100000000000)

    zo_perturb_parameters(args, scaling_factor=1)
    loss1 = model(**inputs).loss

    zo_perturb_parameters(args, scaling_factor=-2)
    loss2 = model(**inputs).loss

    print(loss1 - loss2)
    args.projected_grad = ((loss1 - loss2) / (2 * args.zo_eps)).item()

    zo_perturb_parameters(args, scaling_factor=1)
    
    return (loss1 + loss2) / 2


def set_zo_grad_as_grad(args):
    """
    Update the parameters with the estimated gradients.
    """
    # Reset the random seed for sampling zs
    torch.manual_seed(args.zo_random_seed)     
    for name, param in args.named_parameters_to_optim:
        # Resample z
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        param.grad = args.projected_grad * z


@torch.inference_mode()
def set_grad_with_mask(args, model, inputs, mask_dict):                
    zo_random_seed = np.random.randint(100000000000)
    
    zo_perturb_parameters_with_mask(args, scaling_factor=1, random_seed=zo_random_seed, mask_dict=mask_dict)
    loss1 = model(**inputs).loss

    zo_perturb_parameters_with_mask(args, scaling_factor=-2, random_seed=zo_random_seed, mask_dict=mask_dict)
    loss2 = model(**inputs).loss
    
    zo_perturb_parameters_with_mask(args, scaling_factor=1, random_seed=zo_random_seed, mask_dict=mask_dict)
    
    global_projected_grad = ((loss1 - loss2) / (2 * args.zo_eps)).item()
    torch.manual_seed(zo_random_seed)
    for name, selected_param in args.named_parameters_to_optim.items():
        z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
        selected_param.grad = global_projected_grad * z
    return loss1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wiki2",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="facebook/opt-6.7b"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-7,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--zo_eps",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "--squeezellm",
        action="store_true"
    )
    parser.add_argument(
        "--outlier",
        action="store_true"
    )
    parser.add_argument(
        "--outlier_percentage",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "--random_subset_weights",
        action="store_true"
    )
    parser.add_argument(
        "--grad_mask",
        action="store_true"
    )
    parser.add_argument(
        "--squeezellm_ckpt",
        type=str,
    )
    parser.add_argument(
        "--prefix_tuning",
        action="store_true"
    )
    parser.add_argument(
        "--lora",
        action="store_true"
    )
    parser.add_argument(
        "--FO",
        action="store_true"
    )
    parser.add_argument(
        "--GA",
        type=int,
        default=1e-3
    )
    parser.add_argument(
        "--ntokens",
        type=int,
        default=20
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def eval(model, eval_dataloader, args=None):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        losses.append(loss.repeat(len(batch['input_ids'])))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def main():
    args = parse_args()
    total_batch_size = args.per_device_train_batch_size 

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # Downloading and loading a dataset from the hub.
    if args.dataset == 'wiki2':        
        raw_datasets = load_dataset(
            'wikitext', 'wikitext-2-raw-v1')
    elif args.dataset == 'ptb':
        raw_datasets = load_dataset(
            'ptb-text-only', 'ptb_text_only')

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name)

    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True)

    dtype = torch.float16

    if args.squeezellm:
        from squeezellm_quant import load_quant
        model = load_quant(args.model_name, 
                args.squeezellm_ckpt, 
                4, True, topX=0, 
                sparse_dtype=torch.float16, 
                fake_quant=True,
                use_flash_attn_2=(not args.prefix_tuning))
        
        from squeezellm_quant import load_quant, transfer_quant_linear_to_nn_linear
        if args.prefix_tuning or args.lora:
            transfer_quant_linear_to_nn_linear(model)
            if args.prefix_tuning:
                model.config.use_cache = True
                from prefix import PrefixTuning
                print(f'{args.ntokens} tokens')
                PrefixTuning(model, num_prefix=args.ntokens, reparam=False, float16=True, init_by_real_act=True)

            else:
                from lora import LoRA
                LoRA(model, r=8, alpha=16)

    else:            
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            device_map='auto',
            torch_dtype=dtype,
            attn_implementation='flash_attention_2'
        )

        if args.prefix_tuning:
            model.config.use_cache = True
            from prefix import PrefixTuning
            print(f'{args.ntokens} tokens')
            PrefixTuning(model, num_prefix=args.ntokens, reparam=False, float16=True, init_by_real_act=True)

        elif args.lora:
            from lora import LoRA
            LoRA(model, r=8, alpha=16)

    args.model = model

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    def tokenize_function(examples):
        return tokenizer(['\n\n'.join(examples[text_column_name])])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]

    old_block_size = block_size
    block_size = 128
    train_128_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of 128",
    )["train"]
    block_size = old_block_size


    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    train_128_dataloader = DataLoader(
        train_128_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=1
    )

    if args.squeezellm and (not (args.lora or args.prefix_tuning)):
        args.named_buffers = { name : buffer for name, buffer in model.named_buffers() }
        args.sqllm_sparse_weight_dict = dict()
        args.dequantized_weight_list = [(name, buffer) for name, buffer in model.named_buffers() if 'dequantized_weight' in name]
        for name, buffer in args.dequantized_weight_list:
            indices = args.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
            args.sqllm_sparse_weight_dict[name] = buffer[indices[0], indices[1]].clone().detach().requires_grad_(True)


    has_mask = False
    if args.outlier:
        print(f'outlier mask {args.outlier_percentage}', flush=True)
        sys.stdout.flush()
        outlier_masks = get_outlier_masks(model, args.outlier_percentage)
        has_mask = True

    elif args.random_subset_weights:
        print(f'random mask {args.outlier_percentage}', flush=True)
        sys.stdout.flush()
        outlier_masks = get_random_masks(model, args.outlier_percentage)
        has_mask = True

    elif args.grad_mask:
        print(f'grad mask {args.outlier_percentage}', flush=True)
        sys.stdout.flush()
        outlier_masks = get_gradient_masks(model, train_128_dataloader, args.outlier_percentage)
        has_mask = True
    
    if has_mask:
        args.named_parameters_to_optim = {
            n : p.view(-1)[outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in model.named_parameters() if n in outlier_masks
        }

    if args.squeezellm and (not (args.lora or args.prefix_tuning)):
        print(f'trainable params: {sum(p.numel() for p in args.sqllm_sparse_weight_dict.values())}')
        optimizer = torch.optim.SGD(list(args.sqllm_sparse_weight_dict.values()), lr=args.learning_rate)
    elif has_mask:
        optimizer = torch.optim.SGD(list(args.named_parameters_to_optim.values()), lr=args.learning_rate)
    elif args.FO:
        print(f'trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], 
                                    lr=args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, last_epoch=-1, 
                                                         start_factor=1, end_factor=0, total_iters=args.max_train_steps)

    else:
        print(f'trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        optimizer = torch.optim.SGD([p for n, p in model.named_parameters() if p.requires_grad], 
                                    lr=args.learning_rate)

    overrode_max_train_steps = False

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / 1)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Train!

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # profiler = EventProfiler(torch.device('cuda'))
    
    completed_steps = 0
    grad_acc_steps = 0
    logging_interval = 10
    eval_steps = 200
    cur_loss = 0
    losses = []
    val_perplexities = []
    best_val_ppl = torch.inf
    test_ppl = torch.inf

    # test_ppl = eval(model, test_dataloader, args)
    # print(f"test ppl: {test_ppl:.3f}", flush=True)

    model.eval()
    progress_bar = tqdm(range(args.max_train_steps))

    while completed_steps < args.max_train_steps:
        active_dataloader = train_dataloader

        for step, inputs in enumerate(active_dataloader):
            inputs = {k: v.cuda() for k, v in inputs.items()}
            if args.FO:
                model.train()
                loss = model(**inputs).loss
                cur_loss += loss.detach()
                loss /= args.GA
                loss.backward()
                grad_acc_steps += 1
                if grad_acc_steps > 0 and grad_acc_steps % args.GA == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    grad_acc_steps = 0
                    
            else:
                with torch.inference_mode():
                    if args.squeezellm and (not (args.lora or args.prefix_tuning)):
                        loss = set_squeezellm_sparse_grad(args, model, inputs)
                        
                    elif has_mask:
                        loss = set_grad_with_mask(args, model, inputs, outlier_masks)

                    else:
                        loss = zo_step(args, model, inputs)
                        set_zo_grad_as_grad(args)

                    cur_loss += loss

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if args.squeezellm and (not (args.lora or args.prefix_tuning)):
                        for name, buffer in args.dequantized_weight_list:
                            sensitive_vals = args.sqllm_sparse_weight_dict[name]
                            indices = args.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                            buffer[indices[0], indices[1]] = sensitive_vals

                    if has_mask:
                        for name, param in model.named_parameters():
                            if not name in args.named_parameters_to_optim: continue
                            param.view(-1)[outlier_masks[name]] = args.named_parameters_to_optim[name]

            with torch.inference_mode():
                if args.FO and grad_acc_steps == 0:
                    completed_steps += 1
                    progress_bar.update(1)
                elif not args.FO:
                    completed_steps += 1
                    progress_bar.update(1)                
                
                if completed_steps > 0 and completed_steps % logging_interval == 0:
                    cur_loss /= logging_interval
                    print(f'loss {cur_loss.item():.3f}', flush=True)
                    sys.stdout.flush()
                    losses.append(cur_loss.item())
                    cur_loss.zero_()

                if completed_steps > 0 and completed_steps % eval_steps == 0:
                    val_ppl = eval(model, eval_dataloader, args)
                    val_perplexities.append(val_ppl)
                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        test_ppl = eval(model, test_dataloader, args)

                    print(f"val ppl: {val_ppl:.3f} | test ppl: {test_ppl:.3f}", flush=True)
                    sys.stdout.flush()

                if completed_steps == args.max_train_steps:
                    val_ppl = eval(model, eval_dataloader, args)
                    val_perplexities.append(val_ppl)
                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        test_ppl = eval(model, test_dataloader, args)

                    print(f"test ppl: {test_ppl:.3f}", flush=True)
                    sys.stdout.flush()
                    break

    print('Training finished!')


if __name__ == "__main__":
    main()