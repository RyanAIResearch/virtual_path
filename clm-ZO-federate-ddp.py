# -*- coding: UTF-8 -*-
#
# ProjectName: PromptZO-master
# Description: 
# Author: 
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from torch.utils.data import Dataset, DistributedSampler
from datasets import load_dataset
import sys
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from eventProfiler import EventProfiler
# from data_process.datasets import FewShotDataset
from collections import defaultdict
import transformers
import copy
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForTokenClassification
)
from cvxopt import matrix,solvers
from numpy.random import RandomState
from FederateZO.server import Server
from FederateZO.client import Client
from task import get_task
from dataclasses import dataclass, is_dataclass, asdict
from utils import *
from get_mask import *


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
        "--model_name",
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
        "--iid",
        type=int,
        default=1,
        help="Is iid or noniid, iid is 1, non iid is 0",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=760,
        help="# ICL mode: number of demonstrations; training mode: number of training samples",
    )
    parser.add_argument(
        "--num_dev",
        type=int,
        default=500,
        help="# (only enabled with training) number of development samples",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=1000,
        help="# number of evaluation samples",
    )
    parser.add_argument(
        "--train_set_seed",
        type=int,
        default=2,
        help="# designated seed to sample training samples/demos",
    )
    parser.add_argument(
        "--num_train_sets",
        type=int,
        default=None,
        help="# how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample",
    )
    parser.add_argument(
        "--total_num_users",
        type=int,
        default=10,
        help="total number of users in FL",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=0.5,
        help="alpha in non-iid",
    )
    # LoRA
    parser.add_argument(
        "--lora",
        type=bool,
        default=False,
        help="# whether to use LoRA",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="alpha in lora",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--sfc",
        type=bool,
        default=False,
        help="# whether to use SFC calibration",
    )
    parser.add_argument(
        "--icl_sfc",
        type=bool,
        default=False,
        help="# whether to use SFC calibration for ICL samples",
    )
    parser.add_argument(
        "--dataloader_drop_last",
        type=bool,
        default=False,
        help="Drop the last incomplete batch if it is not divisible by the batch size.",
    )
    parser.add_argument(
        "--remove_unused_columns",
        type=bool,
        default=True,
        help="remove unused columns.",
    )
    parser.add_argument(
        "--has_mask",
        type=bool,
        default=False,
        help="whether use the sparsed mask",
    )
    parser.add_argument(
        "--use_adam",
        type=bool,
        default=False,
        help="use_adam ",
    )
    parser.add_argument(
        "--use_momentum",
        type=bool,
        default=False,
        help="use_momentum ",
    )
    parser.add_argument(
        "--train_as_classification",
        type=bool,
        default=True,
        help="# take the log likelihood of all options and train as classification ",
    )
    parser.add_argument(
        "--smaller_weight_mask",
        type=bool,
        default=False,
        help="smaller_weight_mask",
    )
    parser.add_argument(
        "--outlier",
        type=bool,
        default=False,
        help="outlier",
    )
    parser.add_argument(
        "--random_subset_weights",
        type=bool,
        default=False,
        help="random_subset_weights",
    )
    parser.add_argument(
        "--grad_mask",
        type=bool,
        default=False,
        help="grad_mask",
    )
    parser.add_argument(
        "--C4_grad_mask",
        type=bool,
        default=False,
        help="C4_grad_mask",
    )
    parser.add_argument(
        "--C4_grad_addr",
        type=str,
        default="GRADIENT_CHUNKS_PATH",
        help="C4_grad_mask",
    )
    parser.add_argument(
        "--GraSP_mask",
        type=bool,
        default=False,
        help="GraSP_mask",
    )
    parser.add_argument(
        "--do_train",
        type=bool,
        default=True,
        help="train dataset",
    )
    parser.add_argument(
        "--do_eval",
        type=bool,
        default=True,
        help="eval dataset",
    )
    parser.add_argument(
        "--do_predict",
        type=bool,
        default=True,
        help="test dataset",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="verbose output",
    )
    parser.add_argument(
        "--gpt3_in_context_tail",
        type=bool,
        default=False,
        help="# GPT-3's in-context learning (context at the end)",
    )
    parser.add_argument(
        "--gpt3_in_context_head",
        type=bool,
        default=False,
        help="# GPT-3's in-context learning (context at the beginning)",
    )
    # Generation
    parser.add_argument(
        "--non_diff",
        type=bool,
        default=False,
        help="# use non-differentiable objective (only support F1 for SQuAD for now)",
    )
    parser.add_argument(
        "--auto_demo",
        type=bool,
        default=True,
        help="Automatically generate template for using demonstrations",
    )
    parser.add_argument(
        "--only_train_option",
        type=bool,
        default=True,
        help="whether to only train the option part of the input",
    )
    parser.add_argument(
        "--sampling",
        type=bool,
        default=True,
        help="# whether to use sampling",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="# max number of new tokens to generate",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="# number of beams for generation",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="# top-k for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="# temperature for generation",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="# top-p for generation",
    )
    parser.add_argument(
        "--eos_token",
        type=str,
        default="\n",
        help="# end of sentence token",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default="wiki",
        help="task_name",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="# max length the model can take",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=1,
        help="the number of users attend in ML in each global epoch",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/grand/sbi-fair/yideran/FLZO",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="zo",
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
    parser.add_argument("--training_mode", default='weight_full_batch',
                    choices=['loss_single_batch','weight_full_batch', 'loss_full_batch'], type=str)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=300,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=None,
        help="Number of batches loaded in advance by each worker.",
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
        "--data_dir",
        type=str,
        default="/grand/sbi-fair/yideran/FLZO/datasets/sst2/data",
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
        "--int4",
        action="store_true"
    )
    parser.add_argument(
        "--int8",
        action="store_true"
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
        "--outlier_percentage",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--zo_sample",
        type=int,
        default=4
    )
    parser.add_argument(
        "--ZO_eps",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--use_cache",
        type=bool,
        default=False,
        help="# whether to use cache checkpoint",
    )
    args = parser.parse_args()
    return args


def split_data(args, train_data):
    # iid or non-iid
    if args.iid:
        if args.task_name in ["SST2", "AGNews", "Yelp"]:
            random.shuffle(train_data)

            num_samples_per_client = len(train_data) // args.total_num_users

            client_datasets = []
            for i in range(args.total_num_users):
                start = i * num_samples_per_client
                end = start + num_samples_per_client if i < args.total_num_users - 1 else len(train_data)
                client_datasets.append(train_data[start:end])
            return client_datasets
        else:
            train_dataset = train_data.shuffle()
            num_samples_per_client = len(train_data) // args.total_num_users

            client_datasets = []
            for i in range(args.total_num_users):
                start = i * num_samples_per_client
                end = start + num_samples_per_client if i < args.total_num_users - 1 else len(train_dataset)
                client_datasets.append(train_dataset.select(range(start, end)))
            return client_datasets
    # non-iid
    else:
        if args.task_name in ["SST2", "AGNews", "Yelp"]:
            random.shuffle(train_data)

            client_datasets = Dirichlet_noniid_partition(train_data, args.total_num_users, args.alpha)
            return client_datasets
        else:
            user_dict_train = {}
            rs = RandomState(args.seed)
            user_dict_train, _ = Dirichlet_noniid(train_data,args.num_users,args.alpha_dir,rs)
            return user_dict_train

def Dirichlet_noniid_partition(dataset, num_clients, alpha):
    np.random.seed(42)
    # Step 1: Treat each inner list as a sample and extract labels
    all_samples = dataset  # Do not flatten; keep the small lists intact

    # Extract labels from samples (assuming both dictionaries in a sample have the same label)
    num_classes = 2  # Since SST-2 has labels 0 and 1
    class_indices = defaultdict(list)
    for idx, sample in enumerate(all_samples):
        label = sample[0]['labels']
        class_indices[label].append(idx)
    # Initialize a mapping from client index to list of sample indices
    client_dataidx_map = {i: [] for i in range(num_clients)}

    min_size = 0
    while min_size < 1:
        idx_batch = [[] for _ in range(num_clients)]
        for label in range(num_classes):
            # Get indices for all samples with this label
            indices = class_indices[label]
            np.random.shuffle(indices)
            # Sample Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Balance proportions to ensure no client has too few samples
            proportions = np.array([p * (len(idx_j) < len(all_samples) / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            # Split indices according to the sampled proportions
            split_indices = np.split(indices, proportions)

            # Assign indices to clients
            for j in range(num_clients):
                idx_batch[j].extend(split_indices[j])
        # Find the minimum number of samples assigned to any client
        min_size = min([len(idx_j) for idx_j in idx_batch])
    # Step 4: Create the client datasets as a list
    client_datasets = []
    for client_idx in range(num_clients):
        data_indices = idx_batch[client_idx]
        client_samples = [all_samples[idx] for idx in data_indices]
        client_datasets.append(client_samples)

    return client_datasets


def Dirichlet_noniid(dataset,num_users,alpha,rs):
    """
    Sample dataset with dirichlet distribution and concentration parameter alpha
    """
    # img_num_per_client = len(dataset)//num_users
    dict_users = {i: np.array([],dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset['labels'])
    classes = np.unique(labels)
    num_classes = len(classes)
    labels_idxs = []
    prior_class_distribution = np.zeros(num_classes)
    b = np.zeros(num_classes)
    for i in range(num_classes):
        labels_idxs.append(idxs[labels==classes[i]])
        prior_class_distribution[i] = len(labels_idxs[i])/len(dataset)
        b[i]=len(labels_idxs[i])
    
    data_ratio = np.zeros([num_classes,num_users])

    
    if isinstance(alpha,list):
        for i in range(num_users):
            data_ratio[:,i] = rs.dirichlet(prior_class_distribution*alpha[i])
    else:
        data_ratio = np.transpose(rs.dirichlet(prior_class_distribution*alpha,size=num_users))
    # data_ratio = data_ratio/np.sum(data_ratio,axis=1,keepdims=True)
    # Client_DataSize = len(dataset)//num_users*np.ones([num_users,1],dtype=np.int64)
    print(f'Class_distribution {prior_class_distribution}. Data_ratio {data_ratio}')
    A = matrix(data_ratio)
    b = matrix(b)
    G = matrix(-np.eye(num_users))
    h = matrix(np.zeros([num_users,1]))
    P = matrix(np.eye(num_users))
    q = matrix(np.zeros([num_users,1]))
    try:
        results = solvers.qp(P,q,G,h,A,b)
        Client_DataSize = np.array(results['x'])
        Data_Division = data_ratio*np.transpose(Client_DataSize)
    except ValueError:
        prior_user_distribution = np.array([1/num_users for _ in range(num_users)])
        data_ratio = rs.dirichlet(prior_user_distribution*alpha,size=num_classes)
        Class_DataSize = np.array([int(len(dataset)/num_classes) for _ in range(num_classes)])
        Data_Division = (data_ratio.T*Class_DataSize).T
    # print(Client_DataSize)
    print(Data_Division)
    print(np.sum(Data_Division, axis=0))
    print(np.sum(Data_Division, axis=1))
    rest = []
    for label in range(num_classes):
        for client in range(num_users):
            data_idx = rs.choice(labels_idxs[label],int(Data_Division[label,client]),replace=False)
            dict_users[client] = np.concatenate([dict_users[client],data_idx],0)
            labels_idxs[label] = list(set(labels_idxs[label])-set(data_idx))
        rest = rest+labels_idxs[label]

    rest_clients = rs.choice(range(num_users),len(rest),replace = True)
    
    for n,user in enumerate(rest_clients):
        dict_users[user] = np.append(dict_users[user],rest[n])

    for user in range(num_users):
        rs.shuffle(dict_users[user])
    return dict_users,data_ratio


def main(args):

    ds_config = {
        "train_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",  # 将优化器状态卸载到 CPU
                "pin_memory": True  # 使用锁页内存提高速度
            },
            "offload_param": {
                "device": "cpu",  # 将模型参数卸载到 CPU
                "pin_memory": True
            }
        },
        "zero_force_ds_cpu_optimizer": False,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [args.adam_beta1, args.adam_beta2],
                "weight_decay": args.weight_decay
            }
        }
    }
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator = Accelerator()

    if args.use_cache:
        checkpoint_path = 'checkpoint.pt'
        checkpoint = torch.load(checkpoint_path)
        global_model.load_state_dict(checkpoint['global_model_state_dict'])
        named_parameters_to_optim = checkpoint['named_parameters_to_optim']
        outlier_masks = checkpoint['outlier_masks']
        history_epochs = checkpoint['epochs']
        history_accuracy_values = checkpoint['accuracy_values']
        print(f"successful save checkpoint, from epoch {epochs[-1]+1}")

    # Initialized Model and Tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name:
        config = AutoConfig.from_pretrained(args.model_name)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    config.use_cache = False
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=True)
    elif args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, use_fast=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.add_eos_token = True
    tokenizer.bos_token = '<bos>'
    tokenizer.padding_side = 'right'

    if args.use_cache is False:
        if "roberta" in args.model_name:
            global_model = AutoModelForMaskedLM.from_pretrained(
                args.model_name,
                device_map='auto'
            )
        else:
            print("Use CausalLM")
            global_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map='auto',
                torch_dtype=torch.float16,
                attn_implementation='flash_attention_2'
            )

    if args.lora:
        from lora import LoRA
        LoRA(global_model, r=args.lora_rank, alpha=args.lora_alpha)
        print("execute lora")
    print("task_name: ", args.task_name)
    data_collator = None
    task = None
    # Data Processing SST2 & wiki
    if args.task_name in ["SST2", "AGNews", "Yelp"]:
        set_seed(args.seed)
        task = get_task(args.task_name, args)
        train_samples = None
        dev_samples = None
        eval_samples = None

        train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)
        if args.train_set_seed is not None or args.num_train_sets is not None:
            # Eval samples share one (or multiple) training set(s)
            for train_set_id, train_samples in enumerate(train_sets):
                train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed
                print("num_eval: ", args.num_eval)
                # Sample eval samples
                if args.num_eval is not None:
                    eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                else:
                    eval_samples = task.valid_samples
                if args.trainer != "none":
                    if args.num_dev is not None:
                        # Dev samples
                        dev_samples = train_samples[-args.num_dev:] 
                        train_samples = train_samples[:-args.num_dev]
                    else:
                        dev_samples = None

        train_dataset, eval_dataset, test_dataset = dataset_preprocessing(args, task, tokenizer, train_samples, dev_samples, eval_samples)
 
        if args.only_train_option and not args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            global_model.original_forward = global_model.forward
            global_model.forward = forward_wrap_with_option_len.__get__(global_model, type(global_model))

        if args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification
        data_collator=DataCollatorWithPaddingAndNesting(tokenizer, pad_to_multiple_of=8) if args.train_as_classification else collator(tokenizer, pad_to_multiple_of=8)
    else:
        raw_datasets = load_dataset(
            args.dataset_name, 
            args.dataset_config_name
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

        embedding_size = global_model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            global_model.resize_token_embeddings(len(tokenizer))

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
        
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]
    
    if args.task_name in ["SST2", "AGNews", "Yelp"]:
        client_data_lists = split_data(args, train_dataset.__get__())

    # Split data
    if args.task_name in ["SST2", "AGNews", "Yelp"]:
        # client_data_lists = split_data(args, train_dataset.__get__())

        # global_train_dataloader = get_train_dataloader(args, train_dataset, data_collator, global_model, None, accelerator)
        # eval_dataloader = get_test_dataloader(args, eval_dataset, data_collator, global_model, None, accelerator)
        global_train_dataloader = DataLoader(
            train_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
    else:
        client_data_lists = split_data(args, train_dataset)
        # DataLoaders creation:
        global_train_dataloader = DataLoader(
            train_dataset, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
        )

    if args.adam:
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.learning_rate, 
                                     betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    elif args.momentum:
        optimizer = torch.optim.SGD(global_model.parameters(), 
                                    lr=args.learning_rate, momentum=args.momentum_mu, weight_decay=args.weight_decay)
    elif args.SGD:
        optimizer = torch.optim.SGD(
            global_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()
    print("optimizer: ", optimizer)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(global_train_dataloader) / args.gradient_accumulation_steps)
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

    if args.use_cache is False:
        # Mask Processing
        outlier_masks = None
        if args.smaller_weight_mask:
            print(f'smaller weight mask {args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            outlier_masks = get_smaller_weight_masks(global_model, args.outlier_percentage)
            args.has_mask = True

        elif args.outlier:
            print(f'outlier mask {args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            outlier_masks = get_outlier_masks(global_model, args.outlier_percentage)
            args.has_mask = True

        elif args.random_subset_weights:
            print(f'random mask {args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            outlier_masks = get_random_masks(global_model, args.outlier_percentage)
            args.has_mask = True

        elif args.grad_mask:
            print(f'grad mask {args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            outlier_masks = get_gradient_masks(global_model, global_train_dataloader, args.outlier_percentage)
            args.has_mask = True
        elif args.C4_grad_mask:
            print(f'C4 grad mask {args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            outlier_masks = get_C4_gradient_masks(global_model, args.C4_grad_addr, args.outlier_percentage)
            args.has_mask = True

        elif args.GraSP_mask:
            print(f'GraSP mask {args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            outlier_masks = get_GraSP_mask(global_model, args.outlier_percentage)
            args.has_mask = True

        # the model layer need to update
        named_parameters_to_optim = [(name, param) for name, param in global_model.named_parameters() if param.requires_grad]

        if args.has_mask:
            named_parameters_to_optim = {
                n : p.view(-1)[outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in global_model.named_parameters() if n in outlier_masks
            }

    if args.use_adam:
        if args.has_mask:
            optimizer = torch.optim.Adam(list(named_parameters_to_optim.values()), lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(global_model.parameters(), lr=args.learning_rate)
    
    elif args.use_momentum:
        if args.has_mask:
            optimizer = torch.optim.SGD(list(named_parameters_to_optim.values()), lr=args.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=args.learning_rate, momentum=0.9)

    else:
        if args.has_mask:
            optimizer = torch.optim.SGD(list(named_parameters_to_optim.values()), lr=args.learning_rate)
        else:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=args.learning_rate)

    server = Server(args, global_train_dataloader, eval_dataloader, global_model, outlier_masks, named_parameters_to_optim, optimizer, task, tokenizer)
    import ipdb
    eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
    eval_sampler = DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_samples, sampler=eval_sampler, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    ipdb.set_trace()
    metrics = server.evaluate_accuracy([], eval_samples)
    print("initial accuracy: ", metrics['accuracy'])
    epochs = []
    ppl_values = []
    accuracy_values = []
    # Real time
    if args.training_mode == "loss_single_batch":
        if args.has_mask:
            clients = []
            # Initial Client Local Data
            for client_idx in range(args.total_num_users):
                local_train_data = client_data_lists[client_idx]
                client = Client(args, local_train_data, None, None, outlier_masks, named_parameters_to_optim, data_collator, client_idx)
                clients.append(client)
            if args.use_cache:
                history_last_epoch = history_epochs[-1]
                start_epoch = history_last_epoch + 1 
            else:
                start_epoch = 0  # start training initial step
            # Start Training
            for e in range(start_epoch, args.max_train_steps):
                print(f"Global epoch {e}...")
                client_projected_grad_accumulator = []
                m = max(int(args.frac * args.total_num_users), 1)
                receive_seed_list = server.get_seed_list()
                #sample users for training
                idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
                for idx in idxs_users:
                    client_started = clients[idx]
                    client_model = copy.deepcopy(global_model)
                    client_named_parameters_to_optim = copy.deepcopy(named_parameters_to_optim)
                    client_started.receive_server_info(receive_seed_list, client_model, client_named_parameters_to_optim)
                    grad_client = client_started.local_train()

                    client_projected_grad_accumulator.append(grad_client)
                    print(f"Clinet {idx} finish local training")
                    del client_model
                    del client_named_parameters_to_optim
                    torch.cuda.empty_cache() 
                global_model, named_parameters_to_optim = server.model_aggregate(client_projected_grad_accumulator)
                del client_projected_grad_accumulator
                # Calculate Accuracy
                print("train_set_seed: ", train_set_seed)
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                metrics = server.evaluate_accuracy([], eval_samples) # No in-context learning if there is training
                print("metrics: ", metrics['accuracy'])
                wandb.log({"epoch": e, "accuracy": metrics['accuracy']})
                epochs.append(e)
                # ppl_values.append(ppl)
                accuracy_values.append(metrics['accuracy'])
                # save the checkpoint every 100 epochs
                if (e + 1) % 100 == 0:
                    # save checkpoint，include global_model，named_parameters_to_optim, outlier_masks, epochs array and accuracy_values
                    checkpoint_path = 'checkpoint.pt'
                    checkpoint = {
                        'global_model_state_dict': global_model.state_dict(),
                        'named_parameters_to_optim': named_parameters_to_optim,
                        'outlier_masks': outlier_masks,
                        'epochs': history_epochs + epochs,
                        'accuracy_values': history_accuracy_values + accuracy_values,
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"epoch {e + 1} save checkpoint to {checkpoint_path}")

        else:
            clients = []
            # Initial Client Local Data
            for client_idx in range(args.total_num_users):
                local_train_data = client_data_lists[client_idx]
                client = Client(args, local_train_data, None, None, None, None, data_collator, client_idx)
                clients.append(client)
            # Start Training
            for e in range(args.max_train_steps):
                print(f"Global epoch {e}...")
                client_projected_grad_accumulator = []
                m = max(int(args.frac * args.total_num_users), 1)
                receive_seed_list = server.get_seed_list()
                #sample users for training
                idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
                for idx in idxs_users:
                    client_started = clients[idx]
                    client_model = copy.deepcopy(global_model)
                    client_started.receive_server_info(receive_seed_list, client_model, None)
                    grad_client = client_started.local_train()
                    torch.cuda.empty_cache() 
                    client_projected_grad_accumulator.append(grad_client)
                    print(f"Clinet {idx} finish local training")
                    del client_model
                    torch.cuda.empty_cache() 
                global_model = server.model_aggregate(client_projected_grad_accumulator)
                del client_projected_grad_accumulator
                # ppl = server.model_eval(global_model)
                torch.cuda.empty_cache() 
                # print(f'Current PPL: {ppl:.3f}')
                print("train_set_seed: ", train_set_seed)
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                metrics = server.evaluate_accuracy([], eval_samples) # No in-context learning if there is training
                print("metrics: ", metrics['accuracy'])
                wandb.log({"epoch": e, "accuracy": metrics['accuracy']})
                epochs.append(e)
                # ppl_values.append(ppl)
                accuracy_values.append(metrics['accuracy'])
    elif args.training_mode == "weight_full_batch":
        # Start Training
        for e in range(args.max_train_steps):
            print(f"Global epoch {e}...")
            client_projected_grad_accumulator = []
            m = max(int(args.frac * args.total_num_users), 1)
            #sample users for training
            idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
            for idx in idxs_users:
                client_model = copy.deepcopy(global_model)
                local_train_data = client_data_lists[idx]
                client = Client(args, local_train_data, None, data_collator, idx)
                grad_client=client.local_train()
                client_projected_grad_accumulator.append(grad_client)
                print(f"Clinet {idx} finish local training")
                del client_model
                torch.cuda.empty_cache()
            global_model = server.model_aggregate(client_projected_grad_accumulator)
            del client_projected_grad_accumulator
            ppl = server.model_eval(global_model)
            print(f'Current PPL: {ppl:.3f}')
    elif args.training_mode == "loss_full_batch":
        # Start Training
        for e in range(args.max_train_steps):
            print(f"Global epoch {e}...")
            client_projected_grad_accumulator = []
            m = max(int(args.frac * args.total_num_users), 1)
            receive_seed_list = server.get_seed_list()
            #sample users for training
            idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
            for idx in idxs_users:
                client_model = copy.deepcopy(global_model)
                local_train_data = client_data_lists[idx]
                client = Client(args, local_train_data, receive_seed_list, data_collator, idx)
                grad_client=client.local_train()
                client_projected_grad_accumulator.append(grad_client)
                print(f"Clinet {idx} finish local training")
                del client_model
                torch.cuda.empty_cache() 
            global_model = server.model_aggregate(client_projected_grad_accumulator)
            del client_projected_grad_accumulator
            ppl = server.model_eval(global_model)
            print(f'Current PPL: {ppl:.3f}')
    # if args.grad_mask:
    #     np.savez(f'training_metrics_learning_rate_{args.learning_rate}_outlier_percentage_{args.outlier_percentage}_mask_type_get_grad.npz', epochs=epochs, accuracy=accuracy_values)
    # elif args.C4_grad_mask:
    #     np.savez(f'training_metrics_learning_rate_{args.learning_rate}_outlier_percentage_{args.outlier_percentage}_mask_type_C4_grad_mask.npz', epochs=epochs, accuracy=accuracy_values)




if __name__ == "__main__":
    import wandb
    if dist.get_rank() == 0:
        wandb.init(project='FLZO_Sparse_iid_gemma')
    args = parse_args()
    main(args)