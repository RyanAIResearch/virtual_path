# -*- coding: UTF-8 -*-
#
# ProjectName: PromptZO-master
# Description: 
# Author: 
import os
# os.environ['HF_HOME'] = '/grand/sbi-fair/yideran/HFDataset'
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from torch.utils.data import Dataset
from datasets import load_dataset
import sys
import gc
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
from collections import Counter
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
        default=6,
        help="total number of users in FL",
    )
    # iid_num_users = 4, noniid_num_users = 4
    parser.add_argument(
        "--iid_num_users",
        type=int,
        default=4,
        help="total number of users in FL",
    )
    parser.add_argument(
        "--noniid_num_users",
        type=int,
        default=2,
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
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--training_mode", default='weight_full_batch',
                    choices=['loss_single_batch','weight_full_batch', 'loss_full_batch', 'loss_dynamic_step', 'sparse_dynamic_localstep', 'sparse_dynamic_multiclients', 'sparse_dynamic_earlystppping'], type=str)
    parser.add_argument("--sim_method", default='cos_sim',
                    choices=['cos_sim','inner_product'], type=str)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
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
        "--local_train_steps",
        type=int,
        default=10,
        help="local steps in fl zo client training",
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
    parser.add_argument(
        "--save_accmetrics_path",
        type=str,
        default="/home/yran1/FlZO_Vritual_Path/acc_metrics",
    )
    parser.add_argument(
        "--save_cossim_path",
        type=str,
        default="/home/yran1/FlZO_Vritual_Path/cossim",
    )
    parser.add_argument(
        "--save_norm_path",
        type=str,
        default="/home/yran1/FlZO_Vritual_Path/norm",
    )
    parser.add_argument(
        "--dot_product_path",
        type=str,
        default="/home/yran1/FlZO_Vritual_Path/norm",
    )
    parser.add_argument(
        "--clients_virtual_path",
        type=str,
        default="/home/yran1/FlZO_Vritual_Path/norm",
    )
    parser.add_argument(
        "--selected_client_numbers",
        type=int,
        nargs='+', 
        default=[0]  
    )
    # parser.add_argument(
    #     "--selected_client_number",
    #     type=int,
    #     default=0
    # )
    args = parser.parse_args()
    return args

def split_data_equal_iid_noniid_2class(args, train_data):
    """
    - 如果 noniid_num_users=0，则所有客户端都走 IID 分配；
    - 否则，前 iid_num_users 个客户端走 IID，后 noniid_num_users 个客户端走 Non-IID。
      (适用于二分类场景: num_classes=2)
    """

    # 若在指定任务列表里，则执行分配逻辑；否则直接返回所有数据（兜底逻辑）
    if args.task_name not in ["SST2", "AGNews", "Yelp", "Copa",
                              "BoolQ", "WIC", "CB", "WSC", "ReCoRD", "RTE"]:
        return [train_data]  # 不在任务列表时直接返回

    random.shuffle(train_data)
    total_data_len = len(train_data)
    total_users = args.total_num_users
    assert args.iid_num_users + args.noniid_num_users == total_users, (
        "iid_num_users + noniid_num_users 必须等于 total_num_users。"
    )

    # ========== 场景 1：noniid_num_users = 0 => 全部走 IID ========== 
    if args.noniid_num_users == 0:
        # 直接采用全部 IID 的逻辑：把train_data等分给所有用户
        num_samples_per_client = total_data_len // total_users
        client_datasets = []
        for i in range(total_users):
            start = i * num_samples_per_client
            # 最后一个客户端拿剩余全部，避免数据丢失
            end = start + num_samples_per_client if i < (total_users - 1) else total_data_len
            client_datasets.append(train_data[start:end])
        return client_datasets

    # ========== 场景 2：noniid_num_users > 0 => 前面IID，后面Non-IID ========== 
    # 先把前 iid_num_users 个客户端分成 IID，每个客户端拿到等量数据切片
    num_samples_per_client = total_data_len // total_users  # 用于前iid_num_users个 IID
    client_datasets = []

    # -------- 1) 前 iid_num_users 个客户端 (IID 分配) --------
    for i in range(args.iid_num_users):
        start = i * num_samples_per_client
        # 若是最后一个IID客户端，拿到它这部分的全部数据(以避免数据丢失)
        end = start + num_samples_per_client if i < (args.iid_num_users - 1) else (args.iid_num_users * num_samples_per_client)
        iid_data = train_data[start:end]
        client_datasets.append(iid_data)

    # 剩下的数据 (还没分出去的) 用于后面 noniid_num_users 客户端
    used_iid_samples = args.iid_num_users * num_samples_per_client
    leftover_data = train_data[used_iid_samples:]

    # -------- 2) 后 noniid_num_users 个客户端 (Non-IID 分配) --------
    num_classes = 2  # 仅示例二分类
    # 按类别划分 leftover_data
    class_samples = {
        c: [sample for sample in leftover_data if sample[0]['labels'] == c]
        for c in range(num_classes)
    }

    leftover_len = len(leftover_data)
    # 计算每个 Non-IID 客户端可分到多少数据
    s2 = leftover_len // args.noniid_num_users

    # 检查是否能满足每个 Non-IID 客户端“从单一类别”拿 s2 条数据
    # 如果不够，就递减 s2
    while s2 > 0:
        can_distribute = True
        # 只要保证每个类别都 >= s2 * (需要分配的客户端数 / num_classes)，
        # 这里为了简化，先假定每个非IID客户端只从它对应class_idx拿s2。
        # 只要 class_samples[0] 和 class_samples[1] 都 >= s2，就可分
        for c in range(num_classes):
            if len(class_samples[c]) < s2:
                can_distribute = False
                break
        if can_distribute:
            break
        s2 -= 1

    if s2 == 0:
        raise ValueError("无法满足 Non-IID 分配：剩余数据不足以给这几个客户端分配。")

    # 按照原先的Non-IID逻辑：client i 只拿一个类别的数据
    # class_idx = i % num_classes
    for i in range(args.noniid_num_users):
        class_idx = i % num_classes
        random.shuffle(class_samples[class_idx])
        noniid_data = class_samples[class_idx][:s2]
        # 从后面取非IID数据
        client_datasets.append(noniid_data)
        # 移除已经分配的这块数据
        class_samples[class_idx] = class_samples[class_idx][s2:]

    # 最终返回前(iid_num_users) + 后(noniid_num_users)个客户端数据
    return client_datasets

def split_data(args, train_data):
    # iid or non-iid
    if args.iid:
        if args.task_name in ["SST2", "AGNews", "Yelp", "Copa","BoolQ","WIC","CB","WSC","ReCoRD","RTE"]:

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
        if args.task_name in ["SST2", "AGNews", "Yelp", "Copa","BoolQ","WIC","CB","WSC","ReCoRD","RTE"]:
            random.shuffle(train_data)
            print("Non-IID Split Data")
            client_datasets = Dirichlet_noniid_partition(args, train_data, args.total_num_users, args.alpha)
            return client_datasets
        else:
            user_dict_train = {}
            rs = RandomState(args.seed)
            user_dict_train, _ = Dirichlet_noniid(train_data,args.num_users,args.alpha_dir,rs)
            return user_dict_train

def Dirichlet_noniid_partition(args, dataset, num_clients, alpha):
    np.random.seed(3407)
    # Step 1: Treat each inner list as a sample and extract labels
    all_samples = dataset  # Do not flatten; keep the small lists intact

    # Extract labels from samples (assuming both dictionaries in a sample have the same label)
    if args.task_name == "AGNews":
        num_classes = 4
    else:
        num_classes = 2
    print("task name: ", args.task_name)
    print("num classes: ", num_classes)
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

def move_to_cpu(data):
    if torch.is_tensor(data):
        return data.cpu()
    elif isinstance(data, dict):
        return {k: move_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_cpu(v) for v in data)
    else:
        return data

def main(args):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # accelerator = Accelerator()

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
            # # define the model base dir
            # base_dir = "/grand/sbi-fair/yideran/FlZO_SparseMask"

            # # create the path
            # if "gemma" in args.model_name:
            #     model_dir = os.path.join(base_dir, "google/gemma-2-2b")
            # # elif "qwen" in args.model_name:
            # #     model_dir = os.path.join(base_dir, "Qwen/Qwen2-1.5B")
            # elif "llama" in args.model_name:
            #     model_dir = os.path.join(base_dir, "meta-llama/Llama-3.2-1B")
            # else:
            #     raise ValueError("unknown model name! please check the model name")

            # # create the full checkpoint path
            # checkpoint_dir = f"{model_dir}_{args.task_name}_localtrainepochs_{args.local_train_steps}_iid_{args.iid}"
            # checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint")
            # # check whether the checkpoint exists
            # if os.path.exists(checkpoint_path):
            #     print("Loading from saved checkpoint:", checkpoint_path)
            #     global_model = AutoModelForCausalLM.from_pretrained(
            #         checkpoint_path,
            #         device_map='auto',
            #         torch_dtype=torch.float16,
            #         attn_implementation='flash_attention_2'
            #     )
            # else:
            #     print("checkpoint not found, loading model from", args.model_name)
            #     global_model = AutoModelForCausalLM.from_pretrained(
            #         args.model_name,
            #         device_map='auto',
            #         torch_dtype=torch.float16,
            #         attn_implementation='flash_attention_2'
            #     )

    if args.lora:
        from lora import LoRA
        LoRA(global_model, r=args.lora_rank, alpha=args.lora_alpha)
        print("execute lora")
    print("task_name: ", args.task_name)
    data_collator = None
    task = None
    # Data Processing SST2 & wiki
    if args.task_name in ["SST2", "AGNews", "Yelp", "Copa","BoolQ","WIC","CB","WSC","ReCoRD","RTE","YelpReview"]:
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

    # Split data
    if args.task_name in ["SST2", "AGNews", "Yelp", "Copa","BoolQ","WIC","CB","WSC","ReCoRD","RTE","YelpReview"]:

        if args.training_mode == "sparse_dynamic_localstep" or args.training_mode == "sparse_dynamic_multiclients" or args.training_mode == "sparse_dynamic_earlystppping":
            client_data_lists = split_data_equal_iid_noniid_2class(args, train_dataset.__get__())
        else:
            client_data_lists = split_data(args, train_dataset.__get__())

        # global_train_dataloader = get_train_dataloader(args, train_dataset, data_collator, global_model, None, accelerator)
        # eval_dataloader = get_test_dataloader(args, eval_dataset, data_collator, global_model, None, accelerator)
        global_train_dataloader = DataLoader(
            train_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        def get_label_distribution(client_data_lists):
            label_distributions = []
            for client_id, client_data in enumerate(client_data_lists):
                # 提取每个客户端的数据标签
                labels = [sample[0]['labels'] for sample in client_data]
                # 统计每个标签的频率
                label_counts = Counter(labels)
                label_distributions.append({"client_id": client_id, "distribution": label_counts})
            return label_distributions

        # 假设 client_data_lists 已经生成
        label_distributions = get_label_distribution(client_data_lists)
        for client_dist in label_distributions:
            print(f"Client {client_dist['client_id']} label distribution: {dict(client_dist['distribution'])}")
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
            outlier_masks, c4_pretrain_grad = get_C4_gradient_masks(global_model, args.C4_grad_addr, args.outlier_percentage)
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

    server = Server(args, global_train_dataloader, eval_dataloader, global_model, outlier_masks, named_parameters_to_optim, c4_pretrain_grad, optimizer, task, tokenizer)
    eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
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
                print(f"Memory Allocated1: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                #sample users for training
                idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
                for idx in idxs_users:
                    client_started = clients[idx]
                    client_model = copy.deepcopy(global_model)
                    client_named_parameters_to_optim = copy.deepcopy(named_parameters_to_optim)
                    client_started.receive_server_info(receive_seed_list, client_model, client_named_parameters_to_optim)
                    print(f"Memory Allocated2: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                    grad_client = client_started.local_train()
                    del client_started.local_model
                    del client_started.named_parameters_to_optim
                    client_projected_grad_accumulator.append(grad_client)
                    print(f"Clinet {idx} finish local training")
                    del client_model
                    del client_named_parameters_to_optim
                    gc.collect()
                    torch.cuda.empty_cache()    
                    print(f"Memory Allocated3: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                global_model, named_parameters_to_optim = server.model_aggregate(client_projected_grad_accumulator)
                del client_projected_grad_accumulator
                # Calculate Accuracy
                print("train_set_seed: ", train_set_seed)
                if e % 2 == 0:
                    eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                    metrics = server.evaluate_accuracy([], eval_samples) # No in-context learning if there is training
                    print("metrics: ", metrics['accuracy'])
                    # wandb.log({"epoch": e, "accuracy": metrics['accuracy']})
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Memory Allocated4: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                # epochs.append(e)
                # ppl_values.append(ppl)
                # accuracy_values.append(metrics['accuracy'])
                # save the checkpoint every 100 epochs
                # if (e + 1) % 10 == 0:
                #     # save checkpoint，include global_model，named_parameters_to_optim, outlier_masks, epochs array and accuracy_values
                #     checkpoint_path = 'checkpoint.pt'
                #     checkpoint = {
                #         'global_model_state_dict': global_model.state_dict(),
                #         'named_parameters_to_optim': named_parameters_to_optim,
                #         'outlier_masks': outlier_masks,
                #         'epochs': history_epochs + epochs,
                #         'accuracy_values': history_accuracy_values + accuracy_values,
                #     }
                #     torch.save(checkpoint, checkpoint_path)
                #     print(f"epoch {e + 1} save checkpoint to {checkpoint_path}")

        else:
            clients = []
            # Initial Client Local Data
            for client_idx in range(args.total_num_users):
                local_train_data = client_data_lists[client_idx]
                client = Client(args, local_train_data, None, None, None, None, data_collator, client_idx)
                clients.append(client)
            print(f"Memory Allocated1: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            # Start Training
            for e in range(args.max_train_steps):
                print(f"Global epoch {e}...")
                client_projected_grad_accumulator = []
                m = max(int(args.frac * args.total_num_users), 1)
                receive_seed_list = server.get_seed_list()
                #sample users for training
                idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
                print(f"Memory Allocated2: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                for idx in idxs_users:
                    client_started = clients[idx]
                    client_model = copy.deepcopy(global_model)
                    client_started.receive_server_info(receive_seed_list, client_model, None)
                    grad_client = client_started.local_train()
                    del client_started.local_model
                    client_projected_grad_accumulator.append(grad_client)
                    print(f"Clinet {idx} finish local training")
                    del client_model
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"Memory Allocated3: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                global_model = server.model_aggregate(client_projected_grad_accumulator)
                del client_projected_grad_accumulator
                # ppl = server.model_eval(global_model)
                # print(f'Current PPL: {ppl:.3f}')
                print("train_set_seed: ", train_set_seed)
                gc.collect()
                torch.cuda.empty_cache() 
                # eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                # metrics = server.evaluate_accuracy([], eval_samples) # No in-context learning if there is training
                # print("metrics: ", metrics['accuracy'])
                # wandb.log({"epoch": e, "accuracy": metrics['accuracy']})
                if e % 2 == 0:
                    eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                    metrics = server.evaluate_accuracy([], eval_samples) # No in-context learning if there is training
                    print("metrics: ", metrics['accuracy'])
                print(f"Memory Allocated4: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                    # wandb.log({"epoch": e, "accuracy": metrics['accuracy']})
                # epochs.append(e)
                # ppl_values.append(ppl)
                # accuracy_values.append(metrics['accuracy'])
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
    elif args.training_mode == "loss_dynamic_step":
        # Start Training
        clients = []
        # Initial Client Local Data
        for client_idx in range(args.total_num_users):
            local_train_data = client_data_lists[client_idx]
            client = Client(args, local_train_data, None, None, None, None, data_collator, client_idx)
            clients.append(client)
        print(f"Memory Allocated1: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        # Start Training
        for e in range(args.max_train_steps):
            print(f"Global epoch {e}...")
            client_projected_grad_accumulator = []
            m = max(int(args.frac * args.total_num_users), 1)
            receive_seed_list = server.get_seed_list()
            #sample users for training
            idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
            print(f"Memory Allocated2: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            for idx in idxs_users:
                client_started = clients[idx]
                client_model = copy.deepcopy(global_model)
                client_started.receive_server_info(receive_seed_list, client_model, None)
                grad_client = client_started.local_train()
                del client_started.local_model
                client_projected_grad_accumulator.append(grad_client)
                print(f"Clinet {idx} finish local training")
                del client_model
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Memory Allocated3: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            global_model = server.model_aggregate(client_projected_grad_accumulator, e)
            del client_projected_grad_accumulator
            print("train_set_seed: ", train_set_seed)
            gc.collect()
            torch.cuda.empty_cache() 
            if e % 2 == 0:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                metrics = server.evaluate_accuracy([], eval_samples) # No in-context learning if there is training
                print("metrics: ", metrics['accuracy'])
            print(f"Memory Allocated4: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
    
    elif args.training_mode == "sparse_dynamic_earlystppping":
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
            print(f"Memory Allocated1: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            #sample users for training
            idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
            for idx in idxs_users:
                client_started = clients[idx]
                client_model = copy.deepcopy(global_model)
                client_named_parameters_to_optim = copy.deepcopy(named_parameters_to_optim)
                client_started.receive_server_info(receive_seed_list, client_model, client_named_parameters_to_optim)
                print(f"Memory Allocated2: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                grad_client = client_started.local_train(e)
                del client_started.local_model
                del client_started.named_parameters_to_optim
                client_projected_grad_accumulator.append(grad_client)
                print(f"Clinet {idx} finish local training")
                del client_model
                del client_named_parameters_to_optim
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Memory Allocated3: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            global_model, named_parameters_to_optim = server.model_aggregate(client_projected_grad_accumulator, e)
            del client_projected_grad_accumulator
            # Calculate Accuracy
            if e % 2 == 0:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                metrics = server.evaluate_accuracy([], eval_samples)
                print("metrics: ", metrics['accuracy'])
            # if e == 3:
            #     break
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Memory Allocated4: {torch.cuda.memory_allocated() / 1024 ** 2} MB")

    elif args.training_mode == "sparse_dynamic_multiclients":
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
            print(f"Memory Allocated1: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            #sample users for training
            idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
            for idx in idxs_users:
                client_started = clients[idx]
                client_model = copy.deepcopy(global_model)
                client_named_parameters_to_optim = copy.deepcopy(named_parameters_to_optim)
                client_started.receive_server_info(receive_seed_list, client_model, client_named_parameters_to_optim)
                print(f"Memory Allocated2: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                grad_client = client_started.local_train(e)
                del client_started.local_model
                del client_started.named_parameters_to_optim
                client_projected_grad_accumulator.append(grad_client)
                print(f"Clinet {idx} finish local training")
                del client_model
                del client_named_parameters_to_optim
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Memory Allocated3: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            global_model, named_parameters_to_optim = server.model_aggregate(client_projected_grad_accumulator, e)
            del client_projected_grad_accumulator
            # Calculate Accuracy
            if e % 2 == 0:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                metrics = server.evaluate_accuracy([], eval_samples)
                print("metrics: ", metrics['accuracy'])
            # if e == 3:
            #     break
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Memory Allocated4: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            

    elif args.training_mode == "sparse_dynamic_localstep":
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
            print(f"Memory Allocated1: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            #sample users for training
            idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
            idxs_users = [0]
            for idx in idxs_users:
                client_started = clients[idx]
                client_model = copy.deepcopy(global_model)
                client_named_parameters_to_optim = copy.deepcopy(named_parameters_to_optim)
                client_started.receive_server_info(receive_seed_list, client_model, client_named_parameters_to_optim)
                print(f"Memory Allocated2: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
                grad_client = client_started.local_train()
                del client_started.local_model
                del client_started.named_parameters_to_optim
                client_projected_grad_accumulator.append(grad_client)
                print(f"Clinet {idx} finish local training")
                del client_model
                del client_named_parameters_to_optim
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Memory Allocated3: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            global_model, named_parameters_to_optim = server.model_aggregate(client_projected_grad_accumulator, e)
            del client_projected_grad_accumulator
            # Calculate Accuracy
            if e % 2 == 0:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
                metrics = server.evaluate_accuracy([], eval_samples)
                print("metrics: ", metrics['accuracy'])
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Memory Allocated4: {torch.cuda.memory_allocated() / 1024 ** 2} MB")


if __name__ == "__main__":
    args = parse_args()
    main(args)






























# def split_data_simulate_virtual_path(args, train_data):
#     # 假设多分类任务 (如 2分类、4分类、5分类)
#     # total_num_users=总用户数: iid_num_users=4, noniid_num_users=总类别数 (如5分类则为5)
#     assert args.iid_num_users == 4
#     assert args.total_num_users == args.iid_num_users + args.noniid_num_users

#     total_users = args.total_num_users
#     num_classes = args.noniid_num_users  # Non-IID客户端数量等于类别数

#     random.shuffle(train_data)
#     total_data_len = len(train_data)

#     # 将数据按类划分
#     class_samples = {i: [s for s in train_data if s[0]['labels'] == i] for i in range(num_classes)}

#     # 计算基础配额 s
#     # Non-IID: 每个拿 s 条数据 (每个客户端仅对应一个类别的数据)
#     # IID: 4个，每个 s 条数据，各类别均匀分布
#     # 总需求：IID需 4*s 条数据（每类数据均分到 IID 客户端），Non-IID需 num_classes * s 条数据
#     # 尝试从最大可能的 s 开始
#     s = total_data_len // total_users

#     # 确保所有类别的数据数量 >= s * (num_classes + 4/类别)
#     while s > 0:
#         if all(len(class_samples[i]) >= s for i in range(num_classes)):
#             break
#         s -= 1

#     if s == 0:
#         raise ValueError("无法满足要求：数据不足以平均分配并满足 Non-IID 与 IID 比例要求。")

#     # 确定分配方案
#     # Non-IID client: 每个客户端拿一个类别的 s 条数据
#     noniid_client_datasets = []
#     for i in range(num_classes):
#         random.shuffle(class_samples[i])
#         noniid_client_datasets.append(class_samples[i][:s])
#         class_samples[i] = class_samples[i][s:]  # 移除已分配数据

#     # IID: 4 个客户端，每个需要 s 条数据，均匀分配各类别
#     iid_count = args.iid_num_users
#     iid_client_datasets = [[] for _ in range(iid_count)]

#     # 每类分配到 IID 客户端的配额
#     per_class_quota = (s * iid_count) // num_classes

#     for i in range(num_classes):
#         random.shuffle(class_samples[i])
#         for j in range(iid_count):
#             start = j * per_class_quota // iid_count
#             end = start + per_class_quota // iid_count
#             iid_client_datasets[j].extend(class_samples[i][start:end])

#     client_datasets = iid_client_datasets + noniid_client_datasets
#     return client_datasets
