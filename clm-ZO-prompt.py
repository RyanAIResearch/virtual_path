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
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
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

logger = get_logger(__name__)

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
        default='wikitext-2-raw-v1',
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="facebook/opt-1.3b"
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
        default=16,
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
        default=5000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
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
        default=1024,
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
        "--adam",
        action="store_true",
    )
    parser.add_argument(
        "--momentum",
        action="store_true"
    )
    parser.add_argument(
        "--SGD",
        action="store_true"
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--linear_lr",
        action="store_true"
    )
    parser.add_argument(
        "--cosine_lr",
        action="store_true"
    )
    parser.add_argument(
        "--cosine_lr_with_warmup",
        action="store_true"
    )
    parser.add_argument(
        "--svd",
        action="store_true"
    )
    parser.add_argument(
        "--random_orthonormal",
        action="store_true"
    )
    parser.add_argument(
        "--svd_normalized",
        action="store_true"
    )
    parser.add_argument(
        "--FD",
        action="store_true"
    )
    parser.add_argument(
        "--momentum_mu",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=10
    )
    parser.add_argument(
        "--zo_sample",
        type=int,
        default=4
    )
    parser.add_argument(
        "--ZO_eps",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--use_mlp",
        action="store_true"
    )
    parser.add_argument(
        "--random_init",
        action="store_true"
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def eval(model, eval_dataloader):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(batch)
        loss = outputs.loss
        losses.append(loss.repeat(len(batch['input_ids'])))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    model.train()
    return perplexity


def write_results(args, losses, ppl):
    tag = args.dataset_name + '-' + args.dataset_config_name + '-' + \
        args.model_name_or_path.replace(os.sep, '-') + '-' + \
        str(args.block_size) + "-"
    if args.adam:
        tag += "adam"
    elif args.momentum:
        tag += f"momentum-{args.momentum_mu}"
    elif args.SGD:
        tag += "SGD"
    else:
        raise NotImplementedError()
    
    tag += '-' + str(args.zo_sample) + '-'
    if args.svd:
        tag += "-svd"
    elif args.random_orthonormal:
        tag += "-random_orthonormal"
    elif args.svd_normalized:
        tag += "-svd_normalized"
    
    if args.FD:
        tag += "-FD"
    
    if args.use_mlp:
        tag += "-use_mlp"

    if args.random_init:
        tag += "-random-init"

    tag += f"-{args.per_device_train_batch_size}-ntokens-{args.n_tokens}-{args.lora_rank}-seed-{args.seed}-lr-{args.learning_rate}-eps-{args.ZO_eps}"
    
    torch.save(losses, f'result/prompt-ZO/{tag}-loss.pt')
    torch.save(ppl, f'result/prompt-ZO/{tag}-ppl.pt')



def main():
    args = parse_args()

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
    raw_datasets = load_dataset(
        args.dataset_name, args.dataset_config_name)
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
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    config.use_cache = False
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    sparsity_lookup = {
        0.25: "/home/wg247/lora-crs/sparsegpt/opt-1.3b-0.25-sparsity",
        0.5: "/home/wg247/lora-crs/sparsegpt/opt-1.3b-0.5-sparsity",
        0.75: "/home/wg247/lora-crs/sparsegpt/opt-1.3b-0.75-sparsity",
    }

    if args.adam:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    if args.random_init:
        random_selection = False
    else:
        random_selection = True

    import modeling_opt
    if args.use_mlp:
        model = modeling_opt.OPTPromptTuningLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=dtype,
            n_tokens=args.n_tokens,
            output_dtype=dtype,
            random_selection=random_selection,
            use_mlp=True
        ).cuda()
    else:
        model = modeling_opt.OPTPromptTuningLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=dtype,
            n_tokens=args.n_tokens,
            output_dtype=dtype,
            random_selection=random_selection
        ).cuda()

    # model.soft_prompt = model.soft_prompt.to(torch.float32)
    # model.lm_head = model.lm_head.to(torch.float32)

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

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    
    if args.lora_rank != -1:
        if args.svd:
            U, V = LoRA_more_rank(
                model.soft_prompt.weight.float(), 
                args.lora_rank, 
                U_rand=False
            )
        elif args.svd_normalized:
            U, V = LoRA_more_rank(
                model.soft_prompt.weight.float(), 
                args.lora_rank, 
            )
            # u, s, v = torch.svd(model.get_input_embeddings().weight.float())
            # torch.save((u.cpu(), s.cpu(), v.cpu()), 'model/opt-1.3b-pretrained-embeddings-svd.pt')
            # import pdb; pdb.set_trace()
            if U.shape[1] > model.soft_prompt.weight.shape[0]:
                prompt_token = model.soft_prompt.weight.shape[0]
                # U[:, prompt_token:] /= U[:, prompt_token:].norm(dim=0).unsqueeze(0)
                U[:, prompt_token:] /= math.sqrt(args.lora_rank - prompt_token)

        elif args.random_orthonormal:
            U, V = LoRA_more_rank_random_orthonormal_initialized(
                model.soft_prompt.float(),
                args.lora_rank,
            )
        else:
            U, V = LoRA_more_rank_random_initialized(
                model.soft_prompt.float(), 
                args.lora_rank, 
            )
        U, V = U.to(dtype=model.soft_prompt.dtype), \
            V.to(dtype=model.soft_prompt.dtype)
        model.prompt_U_weight = nn.Parameter(U.clone())
        model.prompt_V_weight = nn.Parameter(V.clone())

        model.soft_prompt.weight.requires_grad_(False)
        model.soft_prompt.weight.data.copy_(
            model.prompt_U_weight.data @ model.prompt_V_weight.data
        )

    ppl = eval(model, eval_dataloader)
    print(f'ppl {ppl:.3f}')

    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(
            args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    elif args.momentum:
        optimizer = torch.optim.SGD(model.parameters(
        ), lr=args.learning_rate, momentum=args.momentum_mu, weight_decay=args.weight_decay)
    elif args.SGD:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()
    print(optimizer)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.linear_lr:
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, 0, args.max_train_steps)

    elif args.cosine_lr:
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, 0, args.max_train_steps)

    elif args.cosine_lr_with_warmup:
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, int(0.1 * args.max_train_steps), args.max_train_steps)

    else:
        lr_scheduler = transformers.get_constant_schedule(optimizer)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
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
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    # profiler = EventProfiler(torch.device('cuda'))
    completed_steps = 0
    logging_interval = 10
    eval_steps = 50
    cur_loss = 0
    losses = []
    ppls = []

    with torch.no_grad():
        while completed_steps < args.max_train_steps:
            for step, batch in enumerate(train_dataloader):
                model.eval()
                batch = {k: v.cuda() for k, v in batch.items()}
                zo_sample = args.zo_sample
                for _ in range(zo_sample):
                    random_vectors = None
                    model, random_vectors = perturb_parameters(
                        model, args.ZO_eps, random_vectors, scaling_factor=1)
                    if args.lora_rank != -1:
                        soft_prompt = model.prompt_U_weight @ model.prompt_V_weight 
                        loss1 = model(batch, soft_prompt=soft_prompt).loss
                    else:
                        loss1 = model(batch).loss

                    model, random_vectors = perturb_parameters(
                        model, args.ZO_eps, random_vectors, scaling_factor=-2)
                    if args.lora_rank != -1:
                        soft_prompt = model.prompt_U_weight @ model.prompt_V_weight 
                        loss2 = model(batch, soft_prompt=soft_prompt).loss
                    else:
                        loss2 = model(batch).loss

                    model, random_vectors = perturb_parameters(
                        model, args.ZO_eps, random_vectors, scaling_factor=1)
                    
                    cur_loss += (loss1 + loss2) / (2 * zo_sample)
                    projected_grad = (loss1 - loss2) / (args.ZO_eps)

                    for n, p in model.named_parameters():
                        if not p.requires_grad:
                            continue
                        g = projected_grad * random_vectors[n]
                        if p.grad is None:
                            p.grad = g.div_(args.zo_sample)
                        else:
                            p.grad.add_(g, alpha=1/args.zo_sample)

                # FD??
                if args.FD and hasattr(model, 'prompt_U_weight'):
                    model.prompt_U_weight.grad.add_(
                       0.00001 * model.prompt_U_weight @ model.prompt_V_weight @ model.prompt_V_weight.T
                    )
                    model.prompt_V_weight.grad.add_(
                       0.00001 * model.prompt_U_weight.T @ model.prompt_U_weight @ model.prompt_V_weight
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if args.lora_rank != -1:
                    model.soft_prompt.weight.data.copy_(model.prompt_U_weight @ model.prompt_V_weight)
                lr_scheduler.step()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps > 0 and completed_steps % logging_interval == 0:
                    cur_loss /= logging_interval
                    print(f'loss {cur_loss.item()}', flush=True)
                    sys.stdout.flush()
                    losses.append(cur_loss.item())
                    cur_loss.zero_()

                if completed_steps > 0 and completed_steps % eval_steps == 0:
                    ppl = eval(model, eval_dataloader)
                    ppls.append(ppl)
                    print(f"ppl: {ppl}", flush=True)
                    sys.stdout.flush()
                    # write_results(args, losses, ppls)

                # if completed_steps > 0 and completed_steps % 100 == 0:
                #     if args.adam:
                #         optimizer_name = 'adam'
                #     elif args.momentum:
                #         optimizer_name = 'momentum'
                #     elif args.SGD:
                #         optimizer_name = 'SGD'
                    
                #     tag = args.model_name_or_path.replace(os.sep, '-')
                #     tag += '-' + str(args.zo_sample) + '-'

                #     if args.svd:
                #         tag += "svd-"
                #     elif args.random_orthonormal:
                #         tag += "random_orthonormal-"
                #     elif args.svd_normalized:
                #         tag += "svd_normalized-"
                    
                #     if args.use_mlp:
                #         tag += "use_mlp-"

                #     if args.FD:
                #         tag += "FD-"
                    
                #     if args.random_init:
                #         tag += "random_init-"

                #     if args.lora_rank != -1:
                #         torch.save({n: p for n, p in model.named_parameters() if p.requires_grad},
                #                    f'model/prompt-ZO/{tag}{optimizer_name}-{args.dataset_name}-{args.dataset_config_name}-step{completed_steps}_model_{args.block_size}-{args.lora_rank}-{args.learning_rate}-{args.ZO_eps}-{args.seed}.pt')
                #     else:
                #         torch.save({n: p for n, p in model.named_parameters() if p.requires_grad},
                #                    f'model/prompt-ZO/{tag}{optimizer_name}-{args.dataset_name}-{args.dataset_config_name}-step{completed_steps}_model_{args.block_size}-{args.learning_rate}-{args.ZO_eps}-{args.seed}.pt')

                if completed_steps >= args.max_train_steps or ppl >= 100:
                    break



if __name__ == "__main__":
    main()
