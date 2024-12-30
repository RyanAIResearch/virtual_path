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
    AutoModelForMaskedLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    DataCollatorForTokenClassification
)
from ZO import *
from cvxopt import matrix,solvers
from numpy.random import RandomState
from FederateZO.server import Server
from FederateZO.client import Client
from task import get_task
from dataclasses import dataclass, is_dataclass, asdict
from utils import *
from get_mask import *
from dataclasses import dataclass, is_dataclass, asdict
from collections import Counter
import numpy as np
import collections
import re
import string

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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
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
        "--task_name",
        type=str,
        default="wiki",
        help="task_name",
    )
    parser.add_argument(
        "--only_train_option",
        type=bool,
        default=True,
        help="whether to only train the option part of the input",
    )
    parser.add_argument(
        "--non_diff",
        type=bool,
        default=False,
        help="# use non-differentiable objective (only support F1 for SQuAD for now)",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=67349,
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
        "--trainer",
        type=str,
        default="zo",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="# max length the model can take",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="# max number of new tokens to generate",
    )
    parser.add_argument(
        "--train_as_classification",
        type=bool,
        default=True,
        help="# take the log likelihood of all options and train as classification ",
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
        "--verbose",
        type=bool,
        default=False,
        help="verbose output",
    )
    args = parser.parse_args()
    return args

@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]

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
    model.train()
    return perplexity


def write_results(args, losses, ppl):
    tag = args.dataset_name + '-' + args.dataset_config_name + '-' + str(args.block_size) + "-"
    if args.adam:
        tag += "adam"
    elif args.momentum:
        tag += "momentum"
    elif args.SGD:
        tag += "SGD"
    else:
        raise NotImplementedError()
    
    tag += f"-seed-{args.seed}-lr-{args.learning_rate}"
    
    torch.save(losses, f'result/adam-vanilla/{tag}-loss.pt')
    torch.save(ppl, f'result/adam-vanilla/{tag}-ppl.pt')


def encode_prompt(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False, generation=False, generation_with_gold=False, max_new_tokens=None):
    
    # Demonstrations for ICL
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    train_prompts = task.train_sep.join(train_prompts).strip()
    
    # sfc or icl_sfc indicates that this example is used for calibration
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc; verbalize_fn = template.verbalize_sfc
    else: 
        encode_fn = template.encode; verbalize_fn = template.verbalize 
            
    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')
    if not generation:
        # We generate one prompt for each candidate (different classes in classification)
        # or different choices in multiple-choice tasks
        verbalized_eval_prompts = [verbalize_fn(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]

        if sfc:
            # Without demonstrations
            final_prompts = verbalized_eval_prompts 
        else:
            # With demonstrations
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
    else:
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            verbalized_eval_prompts = [verbalize_fn(eval_sample, eval_sample.correct_candidate)]
            unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
            option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
        else:
            option_lens = [0]
            final_prompts = [(train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(' ')]

    # Tokenize 
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    # Truncate (left truncate as demonstrations are less important)
    if generation and max_new_tokens is not None:
        max_length = max_length - max_new_tokens

    if any([len(encoding) > max_length for encoding in encodings]):
        print("Exceed max length")
    if tokenizer.bos_token or tokenizer.add_bos_token:
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]  
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]

    return encodings, option_lens

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# calculate the accuracy
def evaluate_accuracy(args, task, tokenizer, model, train_samples, eval_samples, one_train_set_per_eval_sample=False):
    """
    Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
    """
    # Prediction loop
    predictions = []  
    for eval_id, eval_sample in enumerate(eval_samples):
        predictions.append(
            one_step_pred(args, task, tokenizer, model, train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
        )
    # Calculate metrics 
    metric_name = getattr(task, "metric_name", "accuracy")
    metrics = {metric_name: calculate_metric(predictions, metric_name)}
    return metrics

def calculate_metric(predictions, metric_name):
    if metric_name == "accuracy":
        if isinstance(predictions[0].correct_candidate, list):
            return np.mean([pred.predicted_candidate in pred.correct_candidate for pred in predictions])
        else:
            return np.mean([pred.correct_candidate == pred.predicted_candidate for pred in predictions])
    elif metric_name == "em":
        # For question answering
        return np.mean([any([normalize_answer(ans) == normalize_answer(pred.predicted_candidate) for ans in pred.correct_candidate]) for pred in predictions])
    elif metric_name == "f1":
        # For question answering
        f1 = []
        for pred in predictions:
            all_f1s = []
            if pred.correct_candidate[0] == "CANNOTANSWER" or pred.correct_candidate[0] == "no answer":
                f1.append(int(normalize_answer(pred.correct_candidate[0]) == normalize_answer(pred.predicted_candidate)))
            else:
                for ans in pred.correct_candidate:
                    prediction_tokens = normalize_answer(pred.predicted_candidate).split()
                    ground_truth_tokens = normalize_answer(ans).split()
                    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                    num_same = sum(common.values())
                    if num_same == 0:
                        all_f1s.append(0)
                    else:
                        precision = 1.0 * num_same / len(prediction_tokens)
                        recall = 1.0 * num_same / len(ground_truth_tokens)
                        all_f1s.append((2 * precision * recall) / (precision + recall))
                f1.append(max(all_f1s))

        return np.mean(f1)

def one_step_pred(args, task, tokenizer, model, train_samples, eval_sample, verbose=False):
    """
    Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
    """
    verbose = verbose or args.verbose
    # if verbose:
    #     print("========= Example =========")
    #     print(f"Candidate: {eval_sample.candidates}")
    #     print(f"Correct candidate: {eval_sample.correct_candidate}")


    # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
    encoded_candidates, option_lens = encode_prompt(
        task, task.get_template(), train_samples, eval_sample, tokenizer, max_length=args.max_length, 
        generation=task.generation, max_new_tokens=args.max_new_tokens
    )

    # Calibration
    if args.sfc or args.icl_sfc:
        sfc_encoded_candidates, sfc_option_lens = encode_prompt(task, task.get_template(), 
            train_samples, eval_sample, tokenizer, max_length=args.max_length,
            sfc=args.sfc, icl_sfc=args.icl_sfc, generation=task.generation, 
            max_new_tokens=args.max_new_tokens
        )

    outputs = []
    if task.generation:
        # For generation tasks, return the autoregressively-generated text
        output_text = forward(args, model, tokenizer, encoded_candidates[0], generation=True)
        if verbose:
            print("=== Prompt ===")
            print(tokenizer.decode(encoded_candidates[0]))
            print(f"Output: {output_text}") 
        return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
    else:
        # For classification/multiple-choice, calculate the probabilities of all candidates
        for candidate_id, encoded_candidate in enumerate(encoded_candidates):
            selected_log_probs = forward(args, model, tokenizer, encoded_candidate, option_len=option_lens[candidate_id])

            if args.sfc or args.icl_sfc:
                sfc_selected_log_probs = forward(args, model, tokenizer, sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                if verbose:
                    print("=== Candidate %d (without context) SFC ===" % candidate_id)
                    print(tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(task.train_sep)[-1])
                    print(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

            outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if args.sfc or args.icl_sfc else None})

        if args.sfc or args.icl_sfc:
            scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
        else:
            # (Default) length-normalized log probabilities
            # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
            scores = [x['log_probs'].mean().item() for x in outputs]

        # if verbose:
        #     print(f"Prediction scores: {scores}")

        if isinstance(eval_sample.correct_candidate, list):
            # For some datasets there are multiple correct answers
            correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
        else:
            correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

        return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

def forward(args, model, tokenizer, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(model.device)
        if generation:
            args = args
            # Autoregressive generation
            outputs = model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[tokenizer.encode(args.eos_token, add_special_tokens=False)[0], tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                model.eval()
                logits = model(input_ids=input_ids).logits
                labels = input_ids[0, 1:]
                logits = logits[0, :-1] 
                log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


def main():
    args = parse_args()
    total_batch_size = args.per_device_train_batch_size * \
        args.gradient_accumulation_steps

    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)

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

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name,
        device_map='auto'
    ).bfloat16()

    task = None
    data_collator = None
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
            model.original_forward = model.forward
            model.forward = forward_wrap_with_option_len.__get__(model, type(model))

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

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

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

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
    # evaluate_accuracy(args, task, tokenizer, train_samples, eval_samples, one_train_set_per_eval_sample=False)
    metrics = evaluate_accuracy(args, task, tokenizer, model, [], eval_samples) # No in-context learning if there is training
    print("metrics: ", metrics['accuracy'])
    # print(f'ppl {ppl:.3f}')

    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                                     betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    elif args.momentum:
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=args.learning_rate, momentum=args.momentum_mu, weight_decay=args.weight_decay)
    elif args.SGD:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
    ppls = []

    while completed_steps < args.max_train_steps:
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch)[0]
            loss /= args.gradient_accumulation_steps
            loss.backward()
            cur_loss += loss.detach()
            if (step > 0 and step % args.gradient_accumulation_steps) == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
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
                    # ppl = eval(model, eval_dataloader)
                    # ppls.append(ppl)
                    # print(f"ppl: {ppl}", flush=True)
                    metrics = evaluate_accuracy(args, task, tokenizer, model, [], eval_samples)
                    print("global metrics: ", metrics)
                    sys.stdout.flush()
                    # write_results(args, losses, ppls)

                if completed_steps >= args.max_train_steps:
                    break



if __name__ == "__main__":
    main()
