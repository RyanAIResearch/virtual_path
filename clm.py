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
        default=32,
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
        "--sofiag",
        action="store_true",
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
        "--lookahead",
        action="store_true"
    )
    parser.add_argument(
        "--signSGD",
        action="store_true"
    )
    parser.add_argument(
        "--accclip",
        action="store_true"
    )
    parser.add_argument(
        "--adagrad",
        action="store_true"
    )
    parser.add_argument(
        "--nesterov",
        action="store_true"
    )
    parser.add_argument(
        "--ZO",
        action="store_true"
    )
    parser.add_argument(
        "--sofia_hessian_update_step",
        type=int,
        default=5
    )
    parser.add_argument(
        "--zero_order_eps",
        type=float,
        default=1e-2
    )
    parser.add_argument(
        "--zero_order_eps_2",
        type=float,
        default=1e-2
    )
    parser.add_argument(
        "--soft_prompt_learning",
        action="store_true"
    )
    parser.add_argument(
        "--int4",
        action="store_true"
    )
    parser.add_argument(
        "--int8",
        action="store_true"
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None
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
        "--linear_eps",
        action="store_true"
    )
    parser.add_argument(
        "--cosine_eps",
        action="store_true"
    )
    parser.add_argument(
        "--inverse_sqrt_eps",
        action="store_true"
    )
    parser.add_argument(
        "--step_eps",
        action="store_true"
    )
    parser.add_argument(
        "--constant_ratio_eps",
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
        "--grad_checkpoint",
        action="store_true"
    )
    parser.add_argument(
        "--batch_scheduler",
        action="store_true"
    )
    parser.add_argument(
        "--fp32_prompt",
        action="store_true"
    )
    parser.add_argument(
        "--zo_sample",
        type=int,
        default=1
    )
    parser.add_argument(
        "--zo_sample_scheduler",
        action="store_true"
    )
    parser.add_argument(
        "--STP_3",
        action="store_true"
    )
    parser.add_argument(
        "--h_eigenvalues",
        action="store_true"
    )
    parser.add_argument(
        "--h_trace",
        action="store_true"
    )
    parser.add_argument(
        "--h_density",
        action="store_true"
    )
    parser.add_argument(
        "--h_before",
        action="store_true"
    )
    parser.add_argument(
        "--h_mid",
        action="store_true"
    )
    parser.add_argument(
        "--h_after",
        action="store_true"
    )
    parser.add_argument(
        "--sec_order",
        action="store_true"
    )
    parser.add_argument(
        "--H_as_std",
        action="store_true"
    )
    parser.add_argument(
        "--perturb_coordinates_abs",
        action="store_true"
    )
    parser.add_argument(
        "--perturb_coordinates_rel",
        action="store_true"
    )
    parser.add_argument(
        "--svd_initial_value",
        action="store_true"
    )
    parser.add_argument(
        "--svd_alternative",
        action="store_true"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=20
    )
    parser.add_argument(
        "--lora_more_rank",
        action="store_true"
    )
    parser.add_argument(
        "--lora_more_rank_normalized",
        action="store_true"
    )
    parser.add_argument(
        "--lora_act",
        action="store_true"
    )
    parser.add_argument(
        "--svd_and_lora",
        action="store_true"
    )
    parser.add_argument(
        "--U_zero_initialized",
        action="store_true"
    )
    parser.add_argument(
        "--V_randn_initialized",
        action="store_true"
    )
    parser.add_argument(
        "--activation_func",
        type=str,
        default="gelu",
        choices=[
            "relu",
            "gelu",
            "selu",
            "tanh"
        ]
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def eval(model, eval_dataloader, args=None):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        if args is not None:
            batch['prompt_size'] = args.n_tokens
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


def write_results(args, losses, perplexities):
    task_name = args.dataset_name + '-' + \
        args.dataset_config_name + '-' + str(args.block_size)
    if args.sofiag:
        tag = f"sofiag-{args.sofia_hessian_update_step}"
    elif args.adam and not args.ZO:
        tag = "adam"
    elif args.momentum and not args.ZO:
        tag = "momentum"
    elif args.SGD and not args.ZO:
        tag = "SGD"
    elif args.ZO:
        tag = "zo"
        if args.soft_prompt_learning:
            tag += f"-prompt-{args.n_tokens}"
        if args.STP_3:
            tag += "-STP_3"

        if args.SGD:
            tag += "-SGD"
        elif args.momentum and (not args.signSGD):
            tag += "-momentum"
        elif args.nesterov:
            tag += "-nesterov"
        elif args.adam:
            tag += f"-adam-{args.adam_beta1}-{args.adam_beta2}"
        elif args.signSGD:
            tag += "-signSGD"
            if args.momentum:
                tag += "-momentum"
        elif args.accclip:
            tag += "-accclip"
        elif args.lookahead:
            tag += "-lookahead"
        elif args.adagrad:
            tag += "-adagrad"
        else:
            raise NotImplementedError()

        if args.zo_sample_scheduler:
            tag += f"-zo-sample-scheduler"
        else:
            tag += f"-zo_sample-{args.zo_sample}"
    else:
        raise NotImplemented
    if args.sparsity is not None:
        tag = f"sparsity-{args.sparsity}" + tag
    if args.int4:
        tag = "int4-quantization" + tag
    if args.int8:
        tag = "int8-quantization" + tag
    seed = args.seed
    lr = args.learning_rate

    if args.cosine_lr:
        tag += "-cosine-lr"
    elif args.linear_lr:
        tag += "-linear-lr"
    elif args.cosine_lr_with_warmup:
        tag += "-cosine-lr-with-warmup"

    if args.batch_scheduler:
        tag += "-batch-scheduler"

    if args.ZO:
        if args.cosine_eps:
            tag += "-cosine-eps"
        elif args.linear_eps:
            tag += "-linear-eps"
        elif args.inverse_sqrt_eps:
            tag += "-inverse-sqrt-eps"
        elif args.step_eps:
            tag += "-step-eps"

    # torch.save(losses, f'result{os.sep}{tag}-{task_name}-{lr}-loss-seed-{seed}.pt')
    # with open(f'result{os.sep}{tag}-{task_name}-{lr}-loss-seed-{seed}.txt', 'w') as f:
    #     for i in losses:
    #         f.writelines(str(i) + '\n')

    # torch.save(perplexities, f'result{os.sep}{tag}-{task_name}-{lr}-ppl-seed-{seed}.pt')
    # with open(f'result{os.sep}{tag}-{task_name}-{lr}-ppl-seed-{seed}.txt', 'w') as f:
    #     for i in perplexities:
    #         f.writelines(str(i) + '\n')


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

    if args.adam or args.adagrad:
        dtype = torch.bfloat16
        # dtype = torch.float32
    else:
        dtype = torch.float16

    if args.soft_prompt_learning:
        import modeling_opt
        model = modeling_opt.OPTPromptTuningLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=dtype,
            n_tokens=args.n_tokens,
            output_dtype=dtype,
            random_selection=True
        ).cuda()
        if args.sparsity is not None:
            sparse_model = AutoModelForCausalLM.from_pretrained(
                sparsity_lookup[args.sparsity],
                config=config,
            ).to(dtype=dtype, device=torch.device('cuda'))
            model_state_dict = model.state_dict()
            for n, p in sparse_model.named_parameters():
                if n in model_state_dict:
                    model_state_dict[n].data.copy_(p.data)
            del sparse_model
        elif args.int4 or args.int8:
            from transformers import BitsAndBytesConfig
            if args.int4:
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    load_in_4bit=True,
                    device_map='auto',
                    torch_dtype=torch.float16,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    ),
                )
            else:
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    load_in_8bit=True,
                    device_map='auto',
                    torch_dtype=torch.float16,
                    quantization_config=BitsAndBytesConfig(
                        load_in_8bit=True,
                    ),
                )

            def replace_4_or_8bit_linear(model, other_model, module_to_not_convert="lm_head"):
                other_model_iterator = other_model.named_children()
                for name, module in model.named_children():
                    try:
                        other_name, other_module = next(other_model_iterator)
                    except:
                        break
                    if len(list(module.children())) > 0:
                        replace_4_or_8bit_linear(
                            module, other_module, module_to_not_convert)

                    if isinstance(module, nn.Linear) and name != module_to_not_convert:
                        model._modules[name] = other_model._modules[other_name]
                return model
            replace_4_or_8bit_linear(model, quantized_model)
            del quantized_model

        if args.fp32_prompt:
            model.soft_prompt = model.soft_prompt.to(torch.float32)
            model.lm_head = model.lm_head.to(torch.float32)
    else:
        if args.sparsity is None:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                config=config,
            ).to(dtype=dtype, device=torch.device('cuda'))
        elif args.int4:
            from transformers import BitsAndBytesConfig
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_in_4bit=True,
                device_map='auto',
                torch_dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                ),
            )
        elif args.int8:
            from transformers import BitsAndBytesConfig
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_in_8bit=True,
                device_map='auto',
                torch_dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                ),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                sparsity_lookup[args.sparsity],
                config=config,
            ).to(dtype=dtype, device=torch.device('cuda'))

    # diff_soft_prompt_weight = ref_soft_prompt_weight - model.soft_prompt.weight

    # import functorch
    # model.eval()
    # fmodel, all_params, buffers = functorch.make_functional_with_buffers(model)
    # prompt_params = all_params[-1]

    # def loss_func_opt(prompt_params, buffers, input_ids, attention_mask, labels):
    #     data_dict = {
    #         'input_ids': input_ids,
    #         'attention_mask': attention_mask,
    #         'labels': labels,
    #         'soft_prompt_weight': prompt_params
    #     }
    #     loss = fmodel(all_params, buffers, data_dict)
    #     return loss

    # def loss_func_opt(prompt_params, input_ids, attention_mask, labels):
    #     data_dict = {
    #         'input_ids': input_ids,
    #         'attention_mask': attention_mask,
    #         'labels': labels,
    #         'soft_prompt_weight': prompt_params
    #     }
    #     loss = fmodel(all_params, buffers, data_dict)
    #     return loss

    # ft_compute_sample_grad = torch.vmap(functorch.grad(loss_func_opt), in_dims=(None, 0, 0, 0))
    # h_func = torch.func.hessian(loss_func_opt)

    if args.grad_checkpoint:
        torch.utils.checkpoint.checkpoint_sequential(model)

    profiler = EventProfiler(torch.device('cuda'))

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
        train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # model.soft_prompt.weight.data.copy_(torch.load(
    #     'model/ZO-momentum/step10000_model_128.pt', map_location='cpu').detach()
    # )
    # ppl = eval(model, eval_dataloader)
    # print(f'ppl {ppl:.3f}')

    # embedding dimension, ReLU(Wx) + b
    # SVD(prompt)
    # U (randn) (0) V
    
    if args.lora_more_rank and args.soft_prompt_learning:
        U, V = LoRA_more_rank(
            model.soft_prompt.weight.float(), 
            args.lora_rank, 
            U_rand=(not args.U_zero_initialized),
            V_rand=(args.V_randn_initialized)
        )
        U, V = U.to(dtype=model.soft_prompt.weight.dtype), \
            V.to(dtype=model.soft_prompt.weight.dtype)
        model.soft_prompt.weight.requires_grad_(False)
        model.prompt_U_weight = nn.Parameter(U.clone())
        model.prompt_V_weight = nn.Parameter(V.clone())
        if args.lora_act:
            act_func = {
                'gelu': torch.nn.functional.gelu,
                'relu': torch.nn.functional.relu,
                'selu': torch.nn.functional.selu,
                'tanh': torch.nn.functional.tanh
            }[args.activation_func]
            model.soft_prompt.weight.data.copy_(
                act_func(model.prompt_U_weight.data) @
                    model.prompt_V_weight.data
            )        
        else:
            model.soft_prompt.weight.data.copy_(
                model.prompt_U_weight.data @ model.prompt_V_weight.data
            )

        ppl = eval(model, eval_dataloader)
        print(f'ppl {ppl:.3f}')
        # if ppl >= 60:
        #     raise RuntimeError("ppl too high")

    if args.lora_more_rank_normalized and args.soft_prompt_learning:
        U, V = LoRA_more_rank_normalized(model.soft_prompt.weight.float(), args.lora_rank)
        U, V = U.to(dtype=model.soft_prompt.weight.dtype), \
            V.to(dtype=model.soft_prompt.weight.dtype)
        model.soft_prompt.weight.requires_grad_(False)
        model.prompt_U_weight = nn.Parameter(U.clone())
        model.prompt_V_weight = nn.Parameter(V.clone())
        model.soft_prompt.weight.data.copy_(
            model.prompt_U_weight.data @ model.prompt_V_weight.data
        )
        ppl = eval(model, eval_dataloader)
        print(f'ppl {ppl:.3f}')

    if args.svd_and_lora and args.soft_prompt_learning:
        device, dtype = model.soft_prompt.weight.device, model.soft_prompt.weight.dtype
        model.prompt_U_weight = nn.Parameter(torch.zeros(args.n_tokens, args.lora_rank - args.n_tokens, device=device, dtype=dtype))
        model.prompt_V_weight = nn.Parameter(torch.zeros(args.lora_rank - args.n_tokens, model.soft_prompt.weight.shape[1], device=device, dtype=dtype))
        ppl = eval(model, eval_dataloader)
        print(f'ppl {ppl:.3f}')

    if args.svd_alternative and args.soft_prompt_learning and \
            (not args.lora_more_rank):
        U_1st, V_1st = torch.load(
            'model/ZO-momentum/UV_1st_svd_10000_checkpoint.pt')
        # U_1st, V_1st = torch.load('model/ZO-adam/UV_1st_svd.pt')
        U_1st, V_1st = U_1st.cuda(), V_1st.cuda()
        model.soft_prompt.weight.requires_grad_(False)
        model.prompt_U_weight = nn.Parameter(U_1st.reshape(10, 10).clone())
        model.prompt_V_weight = nn.Parameter(V_1st.T.reshape(10, 2048).clone())
        model.soft_prompt.weight.data.copy_(
            model.prompt_U_weight.data @ model.prompt_V_weight.data
        )
        ppl = eval(model, eval_dataloader)
        print(f'ppl {ppl:.3f}')

    if args.svd_initial_value and args.soft_prompt_learning and \
            (not args.lora_more_rank):
        U_1st, V_1st = torch.load(
            'model/ZO-momentum/UV_1st_svd_10000_checkpoint.pt')
        # U_1st, V_1st = torch.load('model/ZO-adam/UV_1st_svd.pt')
        U_1st, V_1st = U_1st.cuda(), V_1st.cuda()
        model.soft_prompt.weight.requires_grad_(False)
        model.prompt_U_weight = nn.Parameter(U_1st.reshape(10, 10).clone())
        model.prompt_V_weight = nn.Parameter(V_1st.T.reshape(10, 2048).clone())
        model.soft_prompt.weight.data.copy_(
            model.prompt_U_weight.data @ model.prompt_V_weight.data
        )
        ppl = eval(model, eval_dataloader)
        print(f'ppl {ppl:.3f}')

    if args.ZO and args.soft_prompt_learning and args.perturb_coordinates_abs:
        # important_weight_pos = torch.load('model/adam/greater_005_weight_pos.pt').cuda()
        model.soft_prompt.weight.data.copy_(torch.load(
            'model/ZO-momentum/step10000_model_128.pt'))
        model.soft_prompt.weight.requires_grad_(False)
        # important_weight_pos = torch.load(
        #     'model/adam/greater_01_weight_pos.pt').cuda()
        important_weight_pos = torch.load(
            'model/ZO-momentum/greater_1_weight_pos_10000checkpoint.pt').cuda()
        model.soft_prompt_update_weight = nn.Parameter(
            model.soft_prompt.weight[important_weight_pos].clone())
        model.soft_prompt_update_weight.requires_grad_(True)

    if args.ZO and args.soft_prompt_learning and args.perturb_coordinates_rel:
        important_weight_pos = torch.load(
            'model/adam/greater_20_relative_weight_pos.pt').cuda()
        model.soft_prompt.weight.requires_grad_(False)
        model.soft_prompt_update_weight = nn.Parameter(
            model.soft_prompt.weight[important_weight_pos].clone())
        model.soft_prompt_update_weight.requires_grad_(True)

    # from pyhessian_utils import hessian
    # h = hessian(model, train_dataloader)
    # if args.h_eigenvalues:
    #     if args.h_before:
    #         max_eigen_value_before = h.eigenvalues(maxIter=50)
    #         torch.save(max_eigen_value_before, 'model/prompt_128_max_eigenvalue_pair_before_training.pt')
    #     elif args.h_mid:
    #         model.soft_prompt.weight.data.copy_(torch.load('model/mid_model_128.pt').data)
    #         max_eigen_value_mid = h.eigenvalues(maxIter=50)
    #         torch.save(max_eigen_value_mid, 'model/prompt_128_max_eigenvalue_pair_mid_training.pt')
    #     elif args.h_after:
    #         model.soft_prompt.weight.data.copy_(torch.load('model/ref_model_128.pt').data)
    #         max_eigen_value_ref = h.eigenvalues(maxIter=50)
    #         torch.save(max_eigen_value_ref, 'model/prompt_128_max_eigenvalue_pair_after_training.pt')

    #     raise

    # elif args.h_trace:
    #     if args.h_before:
    #         H_trace_before = h.trace(maxIter=50)
    #         torch.save(H_trace_before, 'model/prompt_128_H_trace_before_training.pt')
    #     elif args.h_mid:
    #         model.soft_prompt.weight.data.copy_(torch.load('model/mid_model_128.pt').data)
    #         H_trace_mid = h.trace(maxIter=50)
    #         torch.save(H_trace_mid, 'model/prompt_128_H_trace_mid_training.pt')
    #     elif args.h_after:
    #         model.soft_prompt.weight.data.copy_(torch.load('model/ref_model_128.pt').data)
    #         H_trace_ref = h.trace(maxIter=50)
    #         torch.save(H_trace_ref, 'model/prompt_128_H_trace_after_training.pt')

    #     raise
    # elif args.h_density:
    #     if args.h_before:
    #         h_density_before = h.density(iter=25, n_v=3)
    #         torch.save(h_density_before, 'model/prompt_128_h_density_before_training.pt')
    #     elif args.h_mid:
    #         model.soft_prompt.weight.data.copy_(torch.load('model/mid_model_128.pt').data)
    #         h_density_mid = h.density(iter=25, n_v=3)
    #         torch.save(h_density_mid, 'model/prompt_128_h_density_mid_training.pt')
    #     elif args.h_after:
    #         model.soft_prompt.weight.data.copy_(torch.load('model/ref_model_128.pt').data)
    #         h_density_after = h.density(iter=25, n_v=3)
    #         torch.save(h_density_after, 'model/prompt_128_h_density_after_training.pt')

    #     raise

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "layer_norm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    if args.svd_alternative:
        if args.adam:
            U_optimizer = torch.optim.Adam([model.prompt_U_weight], lr=args.learning_rate, betas=(
                args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
            V_optimizer = torch.optim.Adam([model.prompt_V_weight], lr=args.learning_rate, betas=(
                args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
        elif args.momentum:
            U_optimizer = torch.optim.SGD(
                [model.prompt_U_weight], lr=args.learning_rate, momentum=args.momentum_mu, weight_decay=args.weight_decay)
            V_optimizer = torch.optim.SGD(
                [model.prompt_V_weight], lr=args.learning_rate, momentum=args.momentum_mu, weight_decay=args.weight_decay)
        else:
            U_optimizer = torch.optim.SGD(
                [model.prompt_U_weight], lr=args.learning_rate, weight_decay=args.weight_decay)
            V_optimizer = torch.optim.SGD(
                [model.prompt_V_weight], lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        if args.sofiag:
            from sofiag import SophiaG
            optimizer = SophiaG(model.parameters(), lr=args.learning_rate,
                                rho=20, weight_decay=0, bs=total_batch_size)
        elif args.adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(
                args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
        elif args.momentum:
            optimizer = torch.optim.SGD(model.parameters(
            ), lr=args.learning_rate, momentum=args.momentum_mu, weight_decay=args.weight_decay)
        elif args.nesterov:
            optimizer = torch.optim.SGD(model.parameters(
            ), lr=args.learning_rate, momentum=args.momentum_mu, nesterov=True, weight_decay=args.weight_decay)
        elif args.SGD or (args.signSGD and not args.momentum):
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.accclip:
            import accclip
            optimizer = accclip.ACClip(
                model.parameters(), lr=args.learning_rate)
        elif args.lookahead:
            import lookahead
            optimizer = lookahead.Lookahead(optimizer=torch.optim.SGD(
                model.parameters(), lr=args.learning_rate))
        elif args.adagrad:
            optimizer = torch.optim.Adagrad(
                model.parameters(), lr=args.learning_rate)
        else:
            raise NotImplementedError()
        print(optimizer)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )

    if args.svd_alternative:
        if args.linear_lr:
            U_lr_scheduler = transformers.get_linear_schedule_with_warmup(
                U_optimizer, 0.05, args.max_train_steps)
            V_lr_scheduler = transformers.get_linear_schedule_with_warmup(
                V_optimizer, 0.05, args.max_train_steps)

        elif args.cosine_lr:
            U_lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                U_optimizer, 0, args.max_train_steps)
            V_lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                V_optimizer, 0, args.max_train_steps)

        elif args.cosine_lr_with_warmup:
            U_lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                U_optimizer, int(0.1 * args.max_train_steps), args.max_train_steps)
            V_lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                V_optimizer, int(0.1 * args.max_train_steps), args.max_train_steps)

        else:
            U_lr_scheduler = transformers.get_constant_schedule(U_optimizer)
            V_lr_scheduler = transformers.get_constant_schedule(V_optimizer)

    else:
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

    if args.ZO:
        if args.cosine_eps:
            from ZO import CosineEpsilonSchedule
            eps_scheduler = CosineEpsilonSchedule(
                args.zero_order_eps, args.max_train_steps)
        elif args.linear_eps:
            from ZO import LinearEpsilonSchedule
            eps_scheduler = LinearEpsilonSchedule(
                args.zero_order_eps, args.max_train_steps)
        elif args.inverse_sqrt_eps:
            from ZO import InverseSquareRootEpsilonSchedule
            eps_scheduler = InverseSquareRootEpsilonSchedule(
                args.zero_order_eps)
        elif args.step_eps:
            from ZO import StepEpsilonSchedule
            eps_scheduler = StepEpsilonSchedule(
                args.zero_order_eps, args.max_train_steps)
        elif args.constant_ratio_eps:
            from ZO import ConstantRatioEpsilonSchedule
            eps_scheduler = ConstantRatioEpsilonSchedule(
                args.zero_order_eps, model)
        else:
            from ZO import ConstantEpsilonSchedule
            eps_scheduler = ConstantEpsilonSchedule(args.zero_order_eps)

        if args.zo_sample_scheduler:
            from ZO import StepZOSampleSchedule
            zo_sample_scheduler = StepZOSampleSchedule(args.max_train_steps)

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
    # profiler = EventProfiler(torch.device('cuda'))
    completed_steps = 0
    logging_interval = 10
    eval_steps = 50
    cur_loss = 0
    losses = []
    perplexities = []
    # torch.save(model.soft_prompt.weight,
    #     f'model/initial_model_{args.block_size}_{args.seed}.pt')

    while completed_steps < args.max_train_steps:
        model.train()
        active_dataloader = train_dataloader
        if args.batch_scheduler:
            if completed_steps <= 3000:
                active_dataloader = DataLoader(
                    train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=16)
            elif completed_steps > 3000 and completed_steps <= 12000:
                active_dataloader = DataLoader(
                    train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=32)
            else:
                active_dataloader = DataLoader(
                    train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=64)

        for step, batch in enumerate(active_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            if args.sofiag:
                if completed_steps % args.sofia_hessian_update_step == 0:
                    with profiler('optimization'):
                        with profiler('optimization_forward'):
                            outputs = model(**batch)
                            loss = outputs.loss
                            logits = outputs.logits
                            samp_dist = torch.distributions.Categorical(
                                logits=logits)
                            y_hat_sample = samp_dist.sample()
                            loss = F.cross_entropy(
                                logits.view(-1, model.config.vocab_size), y_hat_sample.view(-1))
                        with profiler('optimization_backward'):
                            loss.backward(retain_graph=True)
                        with profiler('optimization_update_hessian'):
                            optimizer.cache_gradient_for_update_hessian(
                                args.gradient_accumulation_steps)
                        optimizer.zero_grad(set_to_none=True)
                        lr_scheduler.step()

                    with profiler('forward'):
                        loss = F.cross_entropy(
                            logits.view(-1, model.config.vocab_size), batch['labels'].view(-1).cuda())
                else:
                    if optimizer.has_cache:
                        optimizer.update_hessian(use_cache=True)
                    with profiler('forward'):
                        outputs = model(**batch)
                        loss = outputs.loss

                loss /= args.gradient_accumulation_steps
                with profiler('backward'):
                    loss.backward()
                cur_loss += loss.detach()

                if (step > 0 and step % args.gradient_accumulation_steps) == 0 or \
                        step == len(active_dataloader) - 1:
                    with profiler('optimization'):
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    progress_bar.update(1)
                    completed_steps += 1

                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)

                    if completed_steps >= args.max_train_steps:
                        break

            elif args.ZO and (args.lora_more_rank or args.lora_more_rank_normalized) \
                and args.soft_prompt_learning and (not args.svd_alternative) and (not args.lora_act):
                model.eval()
                with torch.no_grad():
                    eps = eps_scheduler.step()
                    zo_sample = args.zo_sample
                    for _ in range(zo_sample):
                        z1 = torch.normal(mean=0, std=1,
                                          size=model.prompt_U_weight.size(),
                                          device=model.prompt_U_weight.device,
                                          dtype=model.prompt_U_weight.dtype
                                          )
                        z2 = torch.normal(mean=0, std=1,
                                          size=model.prompt_V_weight.size(),
                                          device=model.prompt_V_weight.device,
                                          dtype=model.prompt_V_weight.dtype
                                          )

                        model.soft_prompt.weight.data.copy_(
                            (model.prompt_U_weight.data + eps *
                             z1) @ (model.prompt_V_weight.data + eps * z2)
                        )
                        loss1 = model(batch).loss

                        model.soft_prompt.weight.data.copy_(
                            (model.prompt_U_weight.data - eps *
                             z1) @ (model.prompt_V_weight.data - eps * z2)
                        )
                        loss2 = model(batch).loss

                        cur_loss += (loss1 + loss2) / (2 * zo_sample)

                        projected_grad = (loss1 - loss2) / (2 * eps)
                        g_U = projected_grad * z1
                        if model.prompt_U_weight.grad is None:
                            model.prompt_U_weight.grad = g_U.div_(
                                args.zo_sample)
                        else:
                            model.prompt_U_weight.grad.add_(
                                g_U, alpha=1/args.zo_sample)

                        g_V = projected_grad * z2
                        if model.prompt_V_weight.grad is None:
                            model.prompt_V_weight.grad = g_V.div_(
                                args.zo_sample)
                        else:
                            model.prompt_V_weight.grad.add_(
                                g_V, alpha=1/args.zo_sample)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    model.soft_prompt.weight.data.copy_(
                        model.prompt_U_weight.data @ model.prompt_V_weight.data
                    )

                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    if completed_steps > 0 and completed_steps % 1000 == 0:
                        if args.adam:
                            optimizer_name = 'adam'
                        elif args.momentum:
                            optimizer_name = 'momentum'
                        elif args.SGD:
                            optimizer_name = 'SGD'
                        extra_tag = ""
                        if args.U_zero_initialized:
                            extra_tag += "U-zero-intialized"
                        if args.V_randn_initialized:
                            extra_tag += "V-randn-initialized"
                        if args.lora_more_rank_normalized:
                            extra_tag += "extra-rank-normalized"

                        torch.save((model.prompt_U_weight, model.prompt_V_weight),
                                   f'model/ZO-{optimizer_name}-svd-more-rank/{args.n_tokens}-step{completed_steps}_model_{args.block_size}-{args.lora_rank}-{extra_tag}-{args.seed}.pt')

                    if completed_steps >= args.max_train_steps:
                        break


            elif args.ZO and args.lora_more_rank and args.soft_prompt_learning and (not args.svd_alternative) and (args.lora_act):
                model.eval()
                with torch.no_grad():
                    eps = eps_scheduler.step()
                    zo_sample = args.zo_sample
                    for _ in range(zo_sample):
                        z1 = torch.normal(mean=0, std=1,
                                          size=model.prompt_U_weight.size(),
                                          device=model.prompt_U_weight.device,
                                          dtype=model.prompt_U_weight.dtype
                                          )
                        z2 = torch.normal(mean=0, std=1,
                                          size=model.prompt_V_weight.size(),
                                          device=model.prompt_V_weight.device,
                                          dtype=model.prompt_V_weight.dtype
                                          )
                        model.soft_prompt.weight.data.copy_(
                            act_func(model.prompt_U_weight.data + eps * z1) @
                                (model.prompt_V_weight.data + eps * z2)
                        )
                        loss1 = model(batch).loss

                        model.soft_prompt.weight.data.copy_(
                            act_func(model.prompt_U_weight.data - eps * z1) @
                                (model.prompt_V_weight.data - eps * z2)
                        )
                        loss2 = model(batch).loss

                        cur_loss += (loss1 + loss2) / (2 * zo_sample)

                        projected_grad = (loss1 - loss2) / (2 * eps)
                        g_U = projected_grad * z1
                        if model.prompt_U_weight.grad is None:
                            model.prompt_U_weight.grad = g_U.div_(args.zo_sample)
                        else:
                            model.prompt_U_weight.grad.add_(g_U, alpha=1/args.zo_sample)

                        g_V = projected_grad * z2
                        if model.prompt_V_weight.grad is None:
                            model.prompt_V_weight.grad = g_V.div_(args.zo_sample)
                        else:
                            model.prompt_V_weight.grad.add_(g_V, alpha=1/args.zo_sample)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    model.soft_prompt.weight.data.copy_(
                        act_func(
                            model.prompt_U_weight.data) @ model.prompt_V_weight.data
                    )

                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    if completed_steps > 0 and completed_steps % 1000 == 0:
                        if args.adam:
                            optimizer_name = 'adam'
                        elif args.momentum:
                            optimizer_name = 'momentum'
                        elif args.SGD:
                            optimizer_name = 'SGD'
                        torch.save(model.soft_prompt.weight.data,
                                   f'model/ZO-{optimizer_name}-svd-more-rank/{args.n_tokens}-step{completed_steps}_model_{args.block_size}-{args.learning_rate}-U-w-{args.activation_func}-{args.seed}.pt')

                    if completed_steps >= args.max_train_steps:
                        break

            elif args.ZO and args.svd_and_lora and args.soft_prompt_learning:
                model.eval()
                with torch.no_grad():
                    eps = eps_scheduler.step()
                    zo_sample = args.zo_sample
                    for _ in range(zo_sample):
                        z1 = torch.normal(mean=0, std=1,
                                          size=model.prompt_U_weight.size(),
                                          device=model.prompt_U_weight.device,
                                          dtype=model.prompt_U_weight.dtype
                                          )
                        z2 = torch.normal(mean=0, std=1,
                                          size=model.prompt_V_weight.size(),
                                          device=model.prompt_V_weight.device,
                                          dtype=model.prompt_V_weight.dtype
                                          )
                        z3 = torch.normal(mean=0, std=1,
                                          size=model.soft_prompt.weight.size(),
                                          device=model.soft_prompt.weight.device,
                                          dtype=model.soft_prompt.weight.dtype
                                          )

                        model.soft_prompt.weight.data.copy_(
                            (model.prompt_U_weight.data + z1 * eps) @ \
                                (model.prompt_V_weight.data + z2 * eps) +
                                model.soft_prompt.weight.data * (1 + eps)
                        )
                        loss1 = model(batch).loss

                        model.soft_prompt.weight.data.copy_(
                            (model.prompt_U_weight.data - z1 * eps) @
                            (model.prompt_V_weight.data - z2 * eps) +
                            model.soft_prompt.weight.data * (1 - eps)
                        )
                        loss2 = model(batch).loss

                        cur_loss += (loss1 + loss2) / (2 * zo_sample)

                        projected_grad = (loss1 - loss2) / (2 * eps)
                        zs = [z1, z2, z3]
                        params = [model.prompt_U_weight, model.prompt_V_weight, model.soft_prompt.weight]
                        for j, p in enumerate(params):
                            g = projected_grad * zs[j]
                            if p.grad is None:
                                p.grad = g.div_(args.zo_sample)
                            else:
                                p.grad.add_(g, alpha=1/args.zo_sample)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    model.soft_prompt.weight.data.add_(
                        model.prompt_U_weight.data @ model.prompt_V_weight.data
                    )

                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    if completed_steps > 0 and completed_steps % 1000 == 0:
                        if args.adam:
                            optimizer_name = 'adam'
                        elif args.momentum:
                            optimizer_name = 'momentum'
                        elif args.SGD:
                            optimizer_name = 'SGD'
                        # torch.save(model.soft_prompt.weight.data,
                        #                f'model/ZO-{optimizer_name}-svd/step{completed_steps}_model_{args.block_size}-{args.learning_rate}.pt')

                    if completed_steps >= args.max_train_steps:
                        break


            elif args.ZO and args.svd_alternative and args.soft_prompt_learning:
                model.eval()
                with torch.no_grad():
                    zo_sample = args.zo_sample
                    for _ in range(zo_sample):
                        if completed_steps % 2 == 0:
                            eps1 = args.zero_order_eps
                            z1 = torch.normal(mean=0, std=1,
                                              size=model.prompt_U_weight.size(),
                                              device=model.prompt_U_weight.device,
                                              dtype=model.prompt_U_weight.dtype
                                              )

                            model.soft_prompt.weight.data.copy_(
                                (model.prompt_U_weight.data + eps1 * z1) @
                                model.prompt_V_weight.data
                            )
                            loss1 = model(batch).loss

                            model.soft_prompt.weight.data.copy_(
                                (model.prompt_U_weight.data - eps1 * z1) @
                                model.prompt_V_weight.data
                            )
                            loss2 = model(batch).loss

                            cur_loss += (loss1 + loss2) / (2 * zo_sample)

                            projected_grad = (loss1 - loss2) / (2 * eps1)
                            g_U = projected_grad * z1
                            if model.prompt_U_weight.grad is None:
                                model.prompt_U_weight.grad = g_U.div_(
                                    args.zo_sample)
                            else:
                                model.prompt_U_weight.grad.add_(
                                    g_U, alpha=1/args.zo_sample)

                        else:
                            eps2 = args.zero_order_eps_2
                            z2 = torch.normal(mean=0, std=1,
                                              size=model.prompt_V_weight.size(),
                                              device=model.prompt_V_weight.device,
                                              dtype=model.prompt_V_weight.dtype
                                              )

                            model.soft_prompt.weight.data.copy_(
                                model.prompt_U_weight.data @
                                (model.prompt_V_weight.data + eps2 * z2)
                            )
                            loss1 = model(batch).loss

                            model.soft_prompt.weight.data.copy_(
                                model.prompt_U_weight.data @
                                (model.prompt_V_weight.data - eps2 * z2)
                            )
                            loss2 = model(batch).loss

                            cur_loss += (loss1 + loss2) / (2 * zo_sample)
                            projected_grad = (loss1 - loss2) / (2 * eps2)
                            g_V = projected_grad * z2

                            if model.prompt_V_weight.grad is None:
                                model.prompt_V_weight.grad = g_V.div_(
                                    args.zo_sample)
                            else:
                                model.prompt_V_weight.grad.add_(
                                    g_V, alpha=1/args.zo_sample)

                    if completed_steps % 2 == 0:
                        U_optimizer.step()
                        U_optimizer.zero_grad(set_to_none=True)
                        U_lr_scheduler.step()
                    else:
                        V_optimizer.step()
                        V_optimizer.zero_grad(set_to_none=True)
                        V_lr_scheduler.step()

                    model.soft_prompt.weight.data.copy_(
                        model.prompt_U_weight.data @ model.prompt_V_weight.data
                    )

                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    if completed_steps > 0 and completed_steps % 1000 == 0:
                        if args.adam:
                            optimizer_name = 'adam'
                        elif args.momentum:
                            optimizer_name = 'momentum'
                        elif args.SGD:
                            optimizer_name = 'SGD'
                        # torch.save(model.soft_prompt.weight.data,
                        #                f'model/ZO-{optimizer_name}-svd/step{completed_steps}_model_{args.block_size}-{args.learning_rate}.pt')

                    if completed_steps >= args.max_train_steps:
                        break

            elif args.ZO and args.svd_initial_value and args.soft_prompt_learning:
                model.eval()
                with torch.no_grad():
                    eps = eps_scheduler.step()
                    zo_sample = args.zo_sample
                    for _ in range(zo_sample):
                        z1 = torch.normal(mean=0, std=1,
                                          size=model.prompt_U_weight.size(),
                                          device=model.prompt_U_weight.device,
                                          dtype=model.prompt_U_weight.dtype
                                          )
                        z2 = torch.normal(mean=0, std=1,
                                          size=model.prompt_V_weight.size(),
                                          device=model.prompt_V_weight.device,
                                          dtype=model.prompt_V_weight.dtype
                                          )

                        model.soft_prompt.weight.data.copy_(
                            (model.prompt_U_weight.data + eps *
                             z1) @ (model.prompt_V_weight.data + eps * z2)
                        )
                        loss1 = model(batch).loss

                        model.soft_prompt.weight.data.copy_(
                            (model.prompt_U_weight.data - eps *
                             z1) @ (model.prompt_V_weight.data - eps * z2)
                        )
                        loss2 = model(batch).loss

                        cur_loss += (loss1 + loss2) / (2 * zo_sample)

                        projected_grad = (loss1 - loss2) / (2 * eps)
                        g_U = projected_grad * z1
                        if model.prompt_U_weight.grad is None:
                            model.prompt_U_weight.grad = g_U.div_(
                                args.zo_sample)
                        else:
                            model.prompt_U_weight.grad.add_(
                                g_U, alpha=1/args.zo_sample)

                        g_V = projected_grad * z2
                        if model.prompt_V_weight.grad is None:
                            model.prompt_V_weight.grad = g_V.div_(
                                args.zo_sample)
                        else:
                            model.prompt_V_weight.grad.add_(
                                g_V, alpha=1/args.zo_sample)

                    optimizer.zero_grad(set_to_none=True)
                    model.soft_prompt.weight.data.copy_(
                        model.prompt_U_weight.data @ model.prompt_V_weight.data
                    )

                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    if completed_steps > 0 and completed_steps % 1000 == 0:
                        if args.adam:
                            optimizer_name = 'adam'
                        elif args.momentum:
                            optimizer_name = 'momentum'
                        elif args.SGD:
                            optimizer_name = 'SGD'
                        # torch.save(model.soft_prompt.weight.data,
                        #                f'model/ZO-{optimizer_name}-svd/step{completed_steps}_model_{args.block_size}-{args.learning_rate}.pt')

                    if completed_steps >= args.max_train_steps:
                        break

            elif args.ZO and args.soft_prompt_learning and \
                    (args.perturb_coordinates_abs or args.perturb_coordinates_rel):
                model.eval()
                with torch.no_grad():
                    eps = eps_scheduler.step()
                    zo_sample = args.zo_sample
                    for _ in range(zo_sample):
                        z = torch.normal(mean=0, std=1,
                                         size=model.soft_prompt_update_weight.size(),
                                         device=model.soft_prompt_update_weight.device,
                                         dtype=model.soft_prompt_update_weight.dtype
                                         )
                        model.soft_prompt.weight[important_weight_pos] += eps * z
                        loss1 = model(batch).loss

                        model.soft_prompt.weight[important_weight_pos] -= 2 * eps * z
                        loss2 = model(batch).loss

                        model.soft_prompt.weight[important_weight_pos] += eps * z

                        cur_loss += (loss1 + loss2) / (2 * zo_sample)
                        projected_grad = (loss1 - loss2) / (2 * eps)

                        g = projected_grad * z
                        if model.soft_prompt_update_weight.grad is None:
                            model.soft_prompt_update_weight.grad = g.div_(
                                args.zo_sample)
                        else:
                            model.soft_prompt_update_weight.grad.add_(
                                g, alpha=1/args.zo_sample)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    model.soft_prompt.weight[important_weight_pos] = model.soft_prompt_update_weight.data

                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    # if completed_steps > 0 and completed_steps % 1000 == 0:
                    #     if args.adam:
                    #         optimizer_name = 'adam'
                    #     elif args.momentum:
                    #         optimizer_name = 'momentum'
                    #     elif args.SGD:
                    #         optimizer_name = 'SGD'
                    #     if args.perturb_coordinates_abs:
                    #         torch.save(model.soft_prompt.weight.data,
                    #                    f'model/ZO-{optimizer_name}/step{completed_steps}_model_{args.block_size}-perturb_coordinates_abs-{args.learning_rate}.pt')
                    #     else:
                    #         torch.save(model.soft_prompt.weight.data,
                    #                    f'model/ZO-{optimizer_name}/step{completed_steps}_model_{args.block_size}-perturb_coordinates_rel-{args.learning_rate}.pt')

                    if completed_steps >= args.max_train_steps:
                        break

            # H_as_std
            elif args.ZO and args.H_as_std:
                model.eval()
                with torch.no_grad():
                    eps = args.zero_order_eps

                    if completed_steps == 0:
                        random_vector_std = {
                            n: 1 for n, p in model.named_parameters() if p.requires_grad}
                        random_vector = None
                        model, random_vector = perturb_parameters(
                            model, eps, random_vector, scaling_factor=1)
                        loss1 = model(batch).loss

                        model, random_vector = perturb_parameters(
                            model, eps, random_vector, scaling_factor=-2)
                        loss2 = model(batch).loss

                        model, random_vector = perturb_parameters(
                            model, eps, random_vector, scaling_factor=1)

                        projected_grad = (loss1 - loss2) / (2 * eps)

                        random_vector_std = {
                            n: None for n, p in model.named_parameters() if p.requires_grad}
                        for name, param in model.named_parameters():
                            if not param.requires_grad:
                                continue
                            random_vector_std[name] = (
                                projected_grad * random_vector[name]) ** 2
                            # random_vector_std[name] = torch.abs(projected_grad * random_vector[name])

                    random_vector = {n: torch.clamp(torch.randn_like(p.data) / (torch.sqrt(
                        random_vector_std[n]) + 1e-4), max=1, min=-1) for n, p in model.named_parameters() if p.requires_grad}

                    model, random_vector = perturb_parameters(
                        model, eps, random_vector, scaling_factor=1)
                    loss1 = model(batch).loss

                    model, random_vector = perturb_parameters(
                        model, eps, random_vector, scaling_factor=-2)
                    loss2 = model(batch).loss

                    model, random_vector = perturb_parameters(
                        model, eps, random_vector, scaling_factor=1)

                    cur_loss += (loss1 + loss2) / 2
                    projected_grad = (loss1 - loss2) / (2 * eps)

                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        param.grad = projected_grad * random_vector[name]
                        random_vector_std[name] = 0.999 * \
                            random_vector_std[name] + 0.001 * param.grad ** 2
                        random_vector_std[name] /= (1 -
                                                    0.999 ** completed_steps)
                        # random_vector_std[name] = 0.9 * \
                        #     random_vector_std[name] + 0.1 * torch.abs(param.grad)
                        # random_vector_std[name] /= (1 - 0.9 ** completed_steps)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()

                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    if completed_steps > 0 and completed_steps % 1000 == 0:
                        if args.adam:
                            optimizer_name = 'adam'
                        elif args.momentum:
                            optimizer_name = 'momentum'
                        elif args.SGD:
                            optimizer_name = 'SGD'
                        # torch.save(model.soft_prompt.weight.data,
                        #            f'model/ZO-{optimizer_name}-sec-order/step{completed_steps}_model_{args.block_size}.pt')

                    if completed_steps >= args.max_train_steps:
                        break

            elif args.ZO and args.sec_order:
                model.eval()
                with torch.no_grad():
                    eps1 = args.zero_order_eps
                    eps2 = args.zero_order_eps_2

                    if completed_steps % 5 == 0:
                        eigenvector_hessian_tmp = compute_ZO_Hessian_eigenvector(
                            model, eps2, batch, power_iter=10)
                        print(eigenvector_hessian_tmp)
                        if completed_steps == 0:
                            eigenvector_hessian = {
                                n: p * 10 for n, p in eigenvector_hessian_tmp.items()}
                        else:
                            eigenvector_hessian = {
                                n: eigenvector_hessian[n] * 0.9 + eigenvector_hessian_tmp[n] for n, p in model.named_parameters() if p.requires_grad}
                        print(eigenvector_hessian)
                        # eigenvector_hessian = {n : p * torch.tensor(500, device=p.device, dtype=p.dtype) for n, p in eigenvector_hessian.items() }

                    model, eigenvector_hessian = perturb_parameters(
                        model, eps1, eigenvector_hessian, scaling_factor=1)
                    loss1 = model(batch).loss

                    model, eigenvector_hessian = perturb_parameters(
                        model, eps1, eigenvector_hessian, scaling_factor=-2)
                    loss2 = model(batch).loss

                    model, eigenvector_hessian = perturb_parameters(
                        model, eps1, eigenvector_hessian, scaling_factor=1)

                    cur_loss += (loss1 + loss2) / 2
                    projected_grad = (loss1 - loss2) / (2 * eps1)

                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        param.grad = projected_grad * eigenvector_hessian[name]

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()

                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    if completed_steps >= args.max_train_steps:
                        break

            # elif args.ZO and args.sec_order:
            #     model.eval()
            #     with torch.no_grad():
            #         eps1 = args.zero_order_eps
            #         eps2 = args.zero_order_eps_2

            #         random_vector = None
            #         loss1 = model(**batch).loss
            #         model_param_copy_before_step1 = { n : p.data.clone() for p in model.named_parameters() if p.requires_grad}

            #         model, random_vector = perturb_parameters(model, eps1, random_vector, scaling_factor=1)
            #         loss2 = model(**batch).loss

            #         model, random_vector = perturb_parameters(model, eps1, random_vector, scaling_factor=-2)
            #         loss3 = model(**batch).loss

            #         step_loss = min(loss1, loss2, loss3)
            #         if step_loss == loss1:
            #             model, random_vector = perturb_parameters(model, eps1, random_vector, scaling_factor=-1)
            #         elif step_loss == loss2:
            #             model, random_vector = perturb_parameters(model, eps1, random_vector, scaling_factor=2)
            #         elif step_loss == loss3:
            #             pass

            #         del random_vector
            #         random_vector_1 = None
            #         random_vector_2 = None

            #         model_param_copy_after_step1 = { n : p.data.clone() for p in model.named_parameters() if p.requires_grad}
            #         copy_model_weight(model, model_param_copy_before_step1)

            #         model, random_vector_1 = perturb_parameters(model, eps2, random_vector_1, scaling_factor=1)
            #         loss1_2 = model(**batch).loss

            #         model, random_vector = perturb_parameters(model, eps2, random_vector_1, scaling_factor=-2)
            #         loss2_2 = model(**batch).loss

            #         (loss2_2 - loss1_2) / (2 * eps)
            #         cur_loss += step_loss

            #         completed_steps += 1
            #         progress_bar.update(1)
            #         if completed_steps > 0 and completed_steps % logging_interval == 0:
            #             cur_loss /= logging_interval
            #             print(f'loss {cur_loss.item()}', flush=True)
            #             sys.stdout.flush()
            #             losses.append(cur_loss.item())
            #             cur_loss.zero_()

            #         if completed_steps > 0 and completed_steps % eval_steps == 0:
            #             ppl = eval(model, eval_dataloader)
            #             perplexities.append(ppl)
            #             print(f"ppl: {ppl}", flush=True)
            #             sys.stdout.flush()
            #             write_results(args, losses, perplexities)

            #         if completed_steps >= args.max_train_steps:
            #             break

            elif args.ZO and args.STP_3:
                model.eval()
                with torch.no_grad():
                    eps = eps_scheduler.step()
                    random_vector = None
                    loss1 = model(**batch).loss

                    model, random_vector = perturb_parameters(
                        model, eps, random_vector, scaling_factor=1)
                    loss2 = model(**batch).loss

                    model, random_vector = perturb_parameters(
                        model, eps, random_vector, scaling_factor=-2)
                    loss3 = model(**batch).loss

                    model, random_vector = perturb_parameters(
                        model, eps, random_vector, scaling_factor=1)

                    step_loss = min(loss1, loss2, loss3)
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        if loss1 == step_loss:
                            grad = torch.zeros_like(param)
                        elif loss2 == step_loss:
                            grad = -eps * random_vector[name]
                        elif loss3 == step_loss:
                            grad = eps * random_vector[name]
                        else:
                            raise NotImplementedError()

                        if param.grad is None:
                            param.grad = grad.div_(args.zo_sample)
                        else:
                            param.grad.add_(grad, alpha=1/args.zo_sample)
                        del random_vector[name]

                    cur_loss += step_loss
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    if completed_steps >= args.max_train_steps:
                        break

            elif args.ZO:
                model.eval()
                with torch.no_grad():
                    eps = eps_scheduler.step()
                    zo_sample = args.zo_sample
                    for _ in range(zo_sample):
                        z = torch.normal(mean=0, std=1,
                                          size=model.soft_prompt.weight.size(),
                                          device=model.soft_prompt.weight.device,
                                         dtype=model.soft_prompt.weight.dtype
                                          )

                        model.soft_prompt.weight.data.add_(z, alpha=eps)
                        loss1 = model(batch).loss

                        model.soft_prompt.weight.data.add_(z, alpha=-2 * eps)
                        loss2 = model(batch).loss

                        model.soft_prompt.weight.data.add_(z, alpha=eps)
                        cur_loss += (loss1 + loss2) / (2 * zo_sample)

                        projected_grad = (loss1 - loss2) / (2 * eps)
                        p = model.soft_prompt.weight
                        g = projected_grad * z
                        if p.grad is None:
                            p.grad = g.div_(args.zo_sample)
                        else:
                            p.grad.add_(g, alpha=1/args.zo_sample)

                    # g = model.soft_prompt.weight.grad.clone()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # with torch.enable_grad():
                    #     model(batch).loss.backward()
                    #     g_error = model.soft_prompt.weight.grad - g
                    # optimizer.zero_grad(set_to_none=True)
                    # g_errors.append(g_error.detach().view(-1))
                    completed_steps += 1
                    progress_bar.update(1)
                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        sys.stdout.flush()
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader, args)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)
                        sys.stdout.flush()
                        write_results(args, losses, perplexities)

                    if completed_steps > 0 and completed_steps % 1000 == 0:
                        torch.save({n : p for n, p in model.named_parameters() if p.requires_grad}, 
                        f'model/ZO-momentum/step{completed_steps}-{args.n_tokens}_model_{args.block_size}-{args.learning_rate}.pt')

                    # if completed_steps == 100:
                    #     torch.save(
                    #         g_errors, f'model/ZO-momentum/grad-error/10000checkpoint-eps-{eps}.pt')

            elif args.adam or args.momentum or args.SGD:
                # avg_grad = ft_compute_sample_grad(prompt_params, buffers, batch['input_ids'], batch['attention_mask'], batch['labels'])
                # def loss_func_opt(prompt_params):
                #     data_dict = {
                #         'input_ids': batch['input_ids'],
                #         'attention_mask': batch['attention_mask'],
                #         'labels': batch['labels'],
                #         'soft_prompt_weight': prompt_params
                #     }
                #     del model.soft_prompt.weight
                #     model.soft_prompt.weight = prompt_params
                #     loss = model(data_dict)
                #     return loss

                # from pyhessian import hessian
                # hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)
                # H = torch.autograd.functional.hessian(loss_func_opt, model.soft_prompt.weight)
                # import pdb
                # pdb.set_trace()
                # torch.autograd.grad(torch.autograd.grad(loss_func_opt(model.soft_prompt.weight), model.soft_prompt.weight, create_graph=True), model.soft_prompt.weight)

                # H = h_func(prompt_params, buffers, batch['input_ids'], batch['attention_mask'], batch['labels'])
                #
                with profiler('forward'):
                    outputs = model(batch)
                    loss = outputs.loss
                    loss /= args.gradient_accumulation_steps
                with profiler('backward'):
                    loss.backward()

                cur_loss += loss.detach()
                if (step > 0 and step % args.gradient_accumulation_steps) == 0 or \
                        step == len(active_dataloader) - 1:
                    with profiler('optimization'):
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    progress_bar.update(1)
                    completed_steps += 1

                    if completed_steps > 0 and completed_steps % logging_interval == 0:
                        cur_loss /= logging_interval
                        print(f'loss {cur_loss.item()}', flush=True)
                        losses.append(cur_loss.item())
                        cur_loss.zero_()

                    if completed_steps > 0 and completed_steps % eval_steps == 0:
                        ppl = eval(model, eval_dataloader)
                        perplexities.append(ppl)
                        print(f"ppl: {ppl}", flush=True)

                    if completed_steps == 50:
                        print('save 50')
                        torch.save(
                            model.soft_prompt.weight, f'model/step50_model_{args.block_size}_{args.seed}.pt')

                    elif completed_steps == 100:
                        print('save 100')
                        torch.save(
                            model.soft_prompt.weight, f'model/step100_model_{args.block_size}_{args.seed}.pt')

                    elif completed_steps == 200:
                        print('save 200')
                        torch.save(model.soft_prompt.weight,
                                   f'model/step200_model_{args.block_size}_{args.seed}.pt')

                    elif completed_steps == 300:
                        print('save 300')
                        torch.save(model.soft_prompt.weight,
                                   f'model/step300_model_{args.block_size}_{args.seed}.pt')

                    elif completed_steps == 500:
                        print('save 500')
                        torch.save(model.soft_prompt.weight,
                                   f'model/step500_model_{args.block_size}_{args.seed}.pt')

                    elif completed_steps == 1000:
                        print('save 1000')
                        torch.save(model.soft_prompt.weight,
                                   f'model/step1000_model_{args.block_size}_{args.seed}.pt')
                        break

                    elif completed_steps >= args.max_train_steps:
                        torch.save(model.soft_prompt.weight,
                                   'model/ref_model_1024.pt')
                        break


if __name__ == "__main__":
    main()
