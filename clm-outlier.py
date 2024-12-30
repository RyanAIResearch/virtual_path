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
import json
import logging
import math
import os
import random
from itertools import chain
import sys

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from eventProfiler import EventProfiler
import transformers
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
from ZO import *


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default='wikitext-2-v1',
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="llama2"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
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
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
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
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=5e-3
    )
    parser.add_argument(
        "--outlier",
        action='store_true'
    )
    parser.add_argument(
        "--random_subset",
        action='store_true'
    )
    parser.add_argument(
        "--grad_weight_prod",
        action='store_true'
    )
    parser.add_argument(
        "--hessian_grad_weight_prod",
        action='store_true'
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0
    )
    parser.add_argument(
        "--lora",
        action='store_true'
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def eval(model, eval_dataloader, args=None):
    losses = []
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model(**batch).loss
        losses.append(loss.repeat(len(batch['input_ids'])))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def write_results(args, losses, perplexities):
    task_name = args.dataset_name + '-' + \
        args.dataset_config_name + '-' + str(args.block_size)

    seed = args.seed
    lr = args.learning_rate

    tag = args.model_name_or_path.replace(os.sep, '-')
    if args.outlier:
        tag += f"outlier-{args.percentage}"
    elif args.random_subset:
        tag += f"random-subset-{args.percentage}"
    elif args.grad_weight_prod:
        tag += f"grad-weight-prod-{args.percentage}"
    elif args.hessian_grad_weight_prod:
        tag += f"hessian-grad-weight-prod-{args.percentage}"
    else:
        tag += ""        

    # torch.save(losses, f'result{os.sep}outlier{os.sep}{tag}-{task_name}-{lr}-loss-seed-{seed}.pt')
    # with open(f'result{os.sep}outlier{os.sep}{tag}-{task_name}-{lr}-loss-seed-{seed}.txt', 'w') as f:
    #     for i in losses:
    #         f.writelines(str(i) + '\n')

    # torch.save(perplexities, f'result{os.sep}outlier{os.sep}{tag}-{task_name}-{lr}-ppl-seed-{seed}.pt')
    # with open(f'result{os.sep}outlier{os.sep}{tag}-{task_name}-{lr}-ppl-seed-{seed}.txt', 'w') as f:
    #     for i in perplexities:
    #         f.writelines(str(i) + '\n')


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def main():
    args = parse_args()
    total_batch_size = args.per_device_train_batch_size * \
        args.gradient_accumulation_steps

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
    # import pdb; pdb.set_trace()
    raw_datasets = load_dataset(
        args.dataset_name, 
        args.dataset_config_name,
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{args.validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{args.validation_split_percentage}%:]",
        )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.model_name_or_path == 'llama2':
        model = AutoModelForCausalLM.from_pretrained(
            '/share/desa/nfs02/llama2/llama-2-7b/7b-huggingface',
            device_map='auto',
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            '/share/desa/nfs02/llama2/llama-2-7b/7b-huggingface',
            # padding_side="left",
            truncation=True,
            use_fast=True, 
        )
        # tokenizer.add_special_tokens({
        #     "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
        #     "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
        #     "unk_token": tokenizer.convert_ids_to_tokens(
        #         model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
        #     ),
        # })
        # if tokenizer._pad_token is None:
        #     smart_tokenizer_and_embedding_resize(
        #         special_tokens_dict=dict(pad_token="[PAD]"),
        #         tokenizer=tokenizer,
        #         model=model,
        #     )
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

        dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=dtype,
            device_map='auto'
        )

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

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
            print(
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

    train_dataset = train_dataset.with_format('torch')
    eval_dataset = eval_dataset.with_format('torch')

    if args.outlier or \
        args.random_subset or \
        args.grad_weight_prod or \
            args.hessian_grad_weight_prod:
        for n, p in model.named_parameters():
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n:
                p.requires_grad_(False)
    
    if args.outlier:
        mask_dict = get_outlier_masks(model, args.percentage)
    elif args.random_subset:
        mask_dict = get_random_masks(model, args.percentage)
    elif args.grad_weight_prod:
        mask_dict = get_gradient_weight_product_masks(model, train_dataset, percentile=args.percentage)
    elif args.hessian_grad_weight_prod:
        mask_dict = get_hessian_grad_weight_product_masks(model, train_dataset, percentile=args.percentage)
    else:
        mask_dict = None    

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    if args.outlier or args.random_subset or args.grad_weight_prod or args.hessian_grad_weight_prod:
        tune_param_dict = {n : p[mask_dict[n]].detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.SGD(list(tune_param_dict.values()), lr=args.learning_rate, momentum=0.9)        
        
    else:
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=args.learning_rate, 
                                    momentum=0.9, weight_decay=args.weight_decay)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(0.3 * args.max_train_steps), args.max_train_steps)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
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
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    logging_interval = 10
    eval_steps = 200
    cur_loss = 0
    losses = []
    perplexities = []

    model.eval()
    with torch.no_grad():
        while completed_steps < args.max_train_steps:
            active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                # if completed_steps > 0 and completed_steps % 200 == 0:
                #     if args.grad_weight_prod:
                #         mask_dict = get_gradient_weight_product_masks(model.bfloat16(), train_dataset, percentile=args.percentage)
                #     elif args.hessian_grad_weight_prod:
                #         mask_dict = get_hessian_grad_weight_product_masks(model.bfloat16(), train_dataset, percentile=args.percentage)
                
                batch = {k: v.cuda() for k, v in batch.items()}    
                eps = args.eps
                
                one_trial_seed = np.random.randint(10000000)
                if args.outlier or args.random_subset or args.grad_weight_prod or args.hessian_grad_weight_prod:                    
                    model = zo_perturb_parameters_with_mask(model, eps, one_trial_seed, mask_dict=mask_dict, scaling_factor=1)
                    loss1 = model(**batch).loss

                    model = zo_perturb_parameters_with_mask(model, eps, one_trial_seed, mask_dict=mask_dict, scaling_factor=-2)
                    loss2 = model(**batch).loss

                    model = zo_perturb_parameters_with_mask(model, eps, one_trial_seed, mask_dict=mask_dict, scaling_factor=1)

                    cur_loss += (loss1 + loss2) / (2)
                    projected_grad = (loss1 - loss2) / (2 * eps)

                    torch.manual_seed(one_trial_seed)

                    for n, p in model.named_parameters():
                        if not p.requires_grad: continue
                        if not n in mask_dict: continue
                        selected_param = p[mask_dict[n]]
                        z = torch.normal(mean=0, std=1, size=selected_param.size(), device=p.device, dtype=p.dtype)
                        tune_param_dict[n].grad = projected_grad * z

                else:
                    model = perturb_parameters(model, eps, one_trial_seed, scaling_factor=1)
                    loss1 = model(**batch).loss

                    model = perturb_parameters(model, eps, one_trial_seed, scaling_factor=-2)
                    loss2 = model(**batch).loss

                    model = perturb_parameters(model, eps, one_trial_seed, scaling_factor=1)

                    cur_loss += (loss1 + loss2) / (2)
                    projected_grad = (loss1 - loss2) / (2 * eps)

                    torch.manual_seed(one_trial_seed)
                    for n, p in model.named_parameters():
                        if not p.requires_grad: continue
                        z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                        p.grad = z.mul_(projected_grad)
                
                optimizer.step()
                if args.outlier or args.random_subset or args.grad_weight_prod or args.hessian_grad_weight_prod:
                    for n, p in model.named_parameters():
                        if n not in tune_param_dict: continue
                        p[mask_dict[n]] = tune_param_dict[n]

                optimizer.zero_grad(set_to_none=True)

                completed_steps += 1
                progress_bar.update(1)
                lr_scheduler.step()
                if completed_steps > 0 and completed_steps % logging_interval == 0:
                    cur_loss /= logging_interval
                    print(f'loss {cur_loss.item():.4f}', flush=True)
                    sys.stdout.flush()
                    losses.append(cur_loss.item())
                    cur_loss.zero_()

                if completed_steps > 0 and completed_steps % eval_steps == 0:
                    ppl = eval(model, eval_dataloader)
                    perplexities.append(ppl)
                    print(f"ppl: {ppl:.4f}", flush=True)
                    sys.stdout.flush()
                    # write_results(args, losses, perplexities)
                    if ppl >= 200:
                        break

                if completed_steps >= args.max_train_steps:
                    break


if __name__ == "__main__":
    main()
