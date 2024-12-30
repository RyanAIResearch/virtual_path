import logging
import math
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import time
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
from trainer import OurTrainer
import random
from model_opt_soft_prompt_learning import OPTPromptTuningLM
import os
import wandb
from eventProfiler import EventProfiler
import shutil

os.environ['TMPDIR'] = '~/.tmp'



@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

    # Number of examples
    num_train: int = 1000 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = 1000 # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples

    # Training
    trainer: str = "none" 
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 

    # MeZO
    zo_eps: float = 1e-3 # eps in MeZO

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA


    # Generation
    sampling: bool = True # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)

    n_tokens: int = 2

    use_sgd: bool = False

    optim: str = 'sgd'

    samples: int = 1

    use_adam: bool = False

    use_momentum: bool = False

    prefix_tuning_one_layer: bool = False

    prefix_layer_id: int = None

    per_device_eval_batch_size: int = 8

    logging_steps: int = 10

    outlier: bool = False

    outlier_percentage: float = 0.005

    random_subset_weights: bool = False

    grad_mask: bool = False

    use_squeezellm: bool = False

    squeezellm_ckpt: str = None

    squeezellm_wbits: int = 4

    lora_rank: int = 16

    eval_steps: int = 50

    mask_path: str = None

    other_task_mask: bool = False

    lr_scheduler_type = 'constant'

    save_grad: bool = False

    grad_save_path: str = None

    model_weight_path: str = None

    use_squeezellm_peft: bool = False

    no_flash_attn_2: bool = False

    memory_limit_scenario: bool = False

    use_full_zo_update: bool = False

    dynamic_mask: bool = False

    GraSP_mask: bool = False

    dynamic_mask_step: int = 100

    save_model_addr: str = None

    C4_grad_mask: bool = False

    C4_grad_addr: str = None

    save_steps: int = 200

    eval_steps: int = 200

    smaller_weight_mask: bool = False

    record_time: bool = False

    #FL parameters:
    total_num_users: int = 4

    per_client_in_training_round: int = 2

    per_device_train_batch_size: int = 8

    training_mode = "loss_full_batch"

    frac = 0.5

    zo_sample = 4

    # parser.add_argument("--training_mode", default='weight_full_batch',
    #                 choices=['loss_single_batch','weight_full_batch', 'loss_full_batch'], type=str)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]

    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model"):
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.add_eos_token = True
            
            tokenizer.padding_side = 'right'

            # HF tokenizer bug fix
            if "opt" in self.args.model_name:
                tokenizer.bos_token_id = 0

            if self.args.use_squeezellm:
                # /share/desa/nfs02/SqueezeLLM/sq-opt-1.3b-w4-s50.pt
                # /share/desa/nfs02/SqueezeLLM/sq-opt-6.7b-w4-s50.pt
                # /share/desa/nfs02/SqueezeLLM/sq-xgen-7b-8k-base-w4-s45.pt
                from squeezellm_quant import load_quant
                if self.args.memory_limit_scenario:
                    torch.cuda.empty_cache()
                    total_cuda_mem_in_GB = torch.cuda.mem_get_info(0)[1] /(1024**3)
                    if total_cuda_mem_in_GB > 10.75: # 2080ti 
                        torch.cuda.set_per_process_memory_fraction(10.75/total_cuda_mem_in_GB)
                    model = load_quant(self.args.model_name, 
                                    self.args.squeezellm_ckpt, 
                                    self.args.squeezellm_wbits, True, topX=0,
                                    sparse_dtype=torch.float32,  
                                    fake_quant=False,
                                    use_flash_attn_2=False,
                                    use_cuda=False)
                    model = model.cuda()
                    torch.cuda.empty_cache()

                else:
                    model = load_quant(self.args.model_name, 
                                    self.args.squeezellm_ckpt, 
                                    self.args.squeezellm_wbits, True, topX=0, 
                                    sparse_dtype=torch.float16, 
                                    fake_quant=True,
                                    use_flash_attn_2=(not self.args.no_flash_attn_2))
                model.eval()
                return model, tokenizer
            
            elif self.args.use_squeezellm_peft:
                from squeezellm_quant import load_quant, transfer_quant_linear_to_nn_linear
                model = load_quant(self.args.model_name, 
                                   self.args.squeezellm_ckpt, 
                                   self.args.squeezellm_wbits, 
                                   True, topX=0, fake_quant=True, 
                                   sparse_dtype=torch.float16,
                                   use_flash_attn_2=((not self.args.prefix_tuning) 
                                                     and (not self.args.no_flash_attn_2)))
                model.eval()
                transfer_quant_linear_to_nn_linear(model)

            else:
                config = AutoConfig.from_pretrained(self.args.model_name)
                config.use_cache = False
                # Auto device loading
                torch_dtype = torch.float16
                if self.args.no_flash_attn_2:
                    attn_impl = 'eager'
                else:
                    attn_impl = 'flash_attention_2'

                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map='auto',
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl
                )

        # Prefix tuning/LoRA
        if self.args.prefix_tuning and not self.args.prefix_tuning_one_layer:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)

        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_rank, alpha=self.args.lora_alpha)

        return model, tokenizer


    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)
        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
                labels = input_ids[0, 1:]
                logits = logits[0, :-1] 
                log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        # if verbose:
        #     print("========= Example =========")
        #     print(f"Candidate: {eval_sample.candidates}")
        #     print(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                print("=== Prompt ===")
                print(self.tokenizer.decode(encoded_candidates[0]))
                print(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                # if verbose:
                    # if candidate_id == 0:
                    #     print("=== Candidate %d ===" % candidate_id)
                    #     print(self.tokenizer.decode(encoded_candidate))
                    # else:
                    #     print("=== Candidate %d (without context)===" % candidate_id)
                    #     print(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    # print(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        print("=== Candidate %d (without context) SFC ===" % candidate_id)
                        print(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        print(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
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


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        # if one_train_set_per_eval_sample:
        #     print(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        # else:
        #     print(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )
        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


    def train(self, train_samples, eval_samples, test_samples=None):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]
            
            def __get__(self):
                return self.data


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))  
            if test_samples:
                test_dataset = HFDataset(_convert(test_samples)) 
                eval_data_for_trainer = {"dev": eval_dataset, "test": test_dataset}
                self.args.metric_for_best_model = 'eval_dev_loss'
            else:
                eval_data_for_trainer = eval_dataset
        
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification
        # if (self.args.trainer == "zo" and self.args.perturb != 'none') or self.args.tune_head_and_untie or self.args.trainer == 'zo_svrg':
        self.args.report_to = []
        self.trainer = OurTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset, 
            eval_dataset=eval_data_for_trainer,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
        )
        self.trainer.train_samples = train_samples
        self.trainer.test_samples = test_samples
        self.trainer.task = self.task
        self.trainer.test_forward = self.forward
        if self.args.save_on_interrupt:
            self.trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        # from transformers.trainer_utils import get_last_checkpoint
        # if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
        #     last_checkpoint = get_last_checkpoint(self.args.output_dir)
        # if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
        #     logger.info(
        #         f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
        #         "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        #     )
        # if self.args.resume_from_checkpoint is not None:
        #     last_checkpoint = self.args.resume_from_checkpoint

        self.trainer.train(resume_from_checkpoint=last_checkpoint) 

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            self.trainer.save_model()
        
        # FSDP compatibility
        self.model = self.trainer.model 

        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                print("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag.replace(os.sep, '-')


def main():
    args = parse_args()
    print(args, flush=True)
    set_seed(args.seed)
    task = get_task(args.task_name, args)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)
    # Initialize trainer and load model
    framework = Framework(args, task)
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
                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples, eval_samples)

                if not args.no_eval:
                    metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)
            print("args_no_eval: ", args.no_eval)
            if not args.no_eval:
                print("===== Train set %d =====" % train_set_seed)
                print(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        print(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)

    shutil.rmtree(framework.trainer.state.best_model_checkpoint)


if __name__ == "__main__": 
    main()
