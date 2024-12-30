########## The following part is copied from Transformers' trainer (3.4.0) and later ported to be compatible with v4.4.2 and to support initialization from linear head probing. ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time
from tqdm.auto import tqdm

import transformers
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

import datasets

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))

class Trainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def should_optim(self, name, param):
        return (not self.args.layer_wise_optim or f".{self.state.global_step % self.model.config.num_hidden_layers}." in name) and param.requires_grad

    @torch.no_grad()
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.add_(z, alpha=(scaling_factor * self.args.zero_order_eps))

    @torch.inference_mode()
    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. 
        """
        # What parameters to optimize 
        # 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zero_order_eps)).item()

        # with self.profiler('perturb'):
        self.zo_perturb_parameters(scaling_factor=1)
        
        return (loss1 + loss2) / 2


    def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if self.args.optimize_acc:
            loss, logits = model(**inputs)
            preds = F.softmax(logits, dim=-1)
            acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
            loss = -acc
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.state.zo_forward_step += 1
        return loss.detach()

    def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps
        return model

    def norm_perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                random_vector[name] = z

            cname = self.retrieve_c(name)
            if cname in self.cs:
                z = z / self.cs[cname]

            param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector
    
    def perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                random_vector[name] = z
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector

    def perturb_single_layer(self, model, layer_name, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            cname = self.retrieve_c(name)
            if cname == layer_name:
                if name in random_vector:
                    z = random_vector[name]
                else:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    random_vector[name] = z
                param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector

    def initialize_c(self, model, inputs):
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if self.should_optim(name, param):
                self.named_parameters_to_optim.append((name, param))

        self.cs = {'embed': 0.0, 'lm_head': 0.0} 
        # OPT: embed_tokens; embed_positions
        # RoBERTa: embeddings
        self.num_params = copy.deepcopy(self.cs)
        self.num_model_layers = model.config.num_hidden_layers
        layer_name = "layers" if model.config.model_type == "opt" else "layer"
        for i in range(self.num_model_layers): 
            self.cs[f'{layer_name}.{i}.'] = 0.0
            self.num_params[f'{layer_name}.{i}.'] = 0
        
        # ZO estimation of c's
        if self.args.zo_variant != 'param_norm' and self.args.use_zo_grad_est:
            for layer in self.cs.keys():
                with torch.no_grad():
                    model, z = self.perturb_single_layer(model, layer_name=layer)
                    loss1 = self.zo_forward(model, inputs)
                    model, z = self.perturb_single_layer(model, layer_name=layer, random_vector=z, scaling_factor=-2)
                    loss2 = self.zo_forward(model, inputs)

                projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)
                self.cs[layer] = torch.abs(projected_grad)

                model, z = self.perturb_single_layer(model, layer_name=layer, random_vector=z)
        
        # no need to run backprop if we are using parameter norm variant, can just measure them
        elif self.args.zo_variant == 'param_norm':
            for name, param in self.named_parameters_to_optim:
                print(name)
                ckey = self.retrieve_c(name)
                if ckey in self.cs:
                    self.cs[ckey] += torch.sum(param.data ** 2)
                    self.num_params[ckey] += param.data.numel()

            # take sqrt to get norm
            for ckey in self.cs:
                self.cs[ckey] = torch.sqrt(self.cs[ckey])
                if self.args.scale_norm_by_num_params:
                    self.cs[ckey] /= torch.sqrt(self.cs[ckey])
            
            for ckey in self.cs:
                if self.cs[ckey] != 0:
                    self.cs[ckey] = self.cs[ckey].detach().item()
        
        # backpropagation estimation fo ZO c's
        #   this is mostly for debugging purposes to disentangle the variance from using ZO to estimate c
        #   from the effectiveness of the preconditioners
        else: 
            model.eval()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()
            for name, param in self.named_parameters_to_optim:
                if param.grad is None:
                    print(name)
                else:
                    ckey = self.retrieve_c(name)
                    if ckey in self.cs:
                        self.cs[ckey] += torch.sum(param.grad ** 2)
                        self.num_params[ckey] += param.grad.numel()

            # take sqrt to get norm
            for ckey in self.cs:
                self.cs[ckey] = torch.sqrt(self.cs[ckey])
                if self.args.scale_norm_by_num_params:
                    self.cs[ckey] /= torch.sqrt(self.num_params[ckey])

            for ckey in self.cs:
                if self.cs[ckey] != 0:
                    self.cs[ckey] = self.cs[ckey].detach().item()

        self.layer_names = list(self.cs.keys())
        model.zero_grad()

    def retrieve_c(self, param_name):
        for c_name in self.cs.keys():
            if c_name in param_name:
                return c_name

        return '' # these parameters are likely not being used in the forward pass

    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1 
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError
        # print("Sample %d zs" % (noise_sample_time))

        return noise_sample_time

    @torch.no_grad()
    def get_outlier_masks(self, model, percentile=0.005):
        mask_dict = dict()
        for n, p in model.named_parameters():
            if 'weight' not in n or 'LayerNorm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n or 'classifier' in n or 'pooler' in n: continue            
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
            mask[(-p.abs()).argsort()[:top_cutoff]] = True
            mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]
        return mask_dict


    @torch.no_grad()
    def get_random_masks(self, model, percentile=0.005):
        mask_dict = dict()
        for n, p in model.named_parameters():
            if 'weight' not in n or 'LayerNorm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n or 'classifier' in n or 'pooler' in n: continue            
            top_cutoff = int(p.numel() * percentile)
            random_indices = torch.randperm(p.numel(), device=p.device)[:int(p.numel() * percentile)]
            mask_dict[n] = random_indices.clone()
            torch.cuda.empty_cache()
        return mask_dict

    def get_gradient_masks(self, model: nn.Module,
                                    percentile=5e-3, microbatch=1, minibatch=16):
        assert minibatch % microbatch == 0
        count = 0
        old_batch_size = self._train_batch_size
        self._train_batch_size = microbatch
        for sampled_batch in self.get_train_dataloader():
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
                if not p.requires_grad or p.grad is None: continue
                if 'weight' not in n or 'LayerNorm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n or 'classifier' in n or 'pooler' in n: continue
                per_layer_scores = p.grad ** 2

                top_cutoff = int(p.numel() * percentile)
                mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
                mask[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
                mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]
                p.grad = None

        self._train_batch_size = old_batch_size
        return mask_dict

    @torch.inference_mode()
    def set_grad_with_mask(self, model, inputs, mask_dict, random_seed):
        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)
        loss1 = self.zo_forward(model, inputs)

        self.zo_perturb_parameters_with_mask(scaling_factor=-2, random_seed=random_seed, mask_dict=mask_dict)
        loss2 = self.zo_forward(model, inputs)

        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)

        global_projected_grad = ((loss1 - loss2) / (2 * self.args.zero_order_eps)).item()
        torch.manual_seed(random_seed)
        for name, selected_param in self.named_parameters_to_optim.items():
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            selected_param.grad = global_projected_grad * z
        return loss1

    def get_grad(self, model, inputs, random_seed):
        self.zo_perturb_parameters(scaling_factor=1, random_seed=random_seed)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=-2, random_seed=random_seed)
        loss2 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=1, random_seed=random_seed)
        global_projected_grad = ((loss1 - loss2) / (2 * self.args.zero_order_eps)).item()
        torch.manual_seed(random_seed)
        ret_grad_dict = dict()
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            ret_grad_dict[name] = global_projected_grad * z
        return ret_grad_dict, loss1


    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        model = self.model

        if self.args.use_squeezellm:
            self.named_buffers = { name : buffer for name, buffer in model.named_buffers() }
            self.sqllm_sparse_weight_dict = dict()
            if self.args.memory_limit_scenario:
                self.sqllm_sparse_weight_dict = { name : buffer.requires_grad_(True) for name, buffer in model.named_buffers() if 'sensitive_vals' in name}
            else:
                self.dequantized_weight_list = [(name, buffer) for name, buffer in self.model.named_buffers() if 'dequantized_weight' in name]
                for name, buffer in self.dequantized_weight_list:
                    indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                    self.sqllm_sparse_weight_dict[name] = buffer[indices[0], indices[1]].clone().detach().requires_grad_(True)


        has_mask = False    
        if self.args.outlier:
            print(f'outlier mask {self.args.outlier_percentage}', flush=True)
            self.outlier_masks = self.get_outlier_masks(self.model, self.args.outlier_percentage)
            has_mask = True

        elif self.args.random_subset_weights:
            print(f'random mask {self.args.outlier_percentage}', flush=True)
            self.outlier_masks = self.get_random_masks(self.model, self.args.outlier_percentage)
            has_mask = True

        elif self.args.grad_mask:
            print(f'grad mask {self.args.outlier_percentage}', flush=True)
            self.outlier_masks = self.get_gradient_masks(self.model, self.args.outlier_percentage)
            has_mask = True


        if has_mask:
            self.named_parameters_to_optim = {
                n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in self.model.named_parameters() if n in self.outlier_masks
            }

        if self.args.use_squeezellm:
            self.optimizer = optimizer = torch.optim.SGD(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate)
            self.lr_scheduler = lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(t_total),
                num_training_steps=t_total,
            )
        elif has_mask:
            self.optimizer = optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
            self.lr_scheduler = lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(t_total),
                num_training_steps=t_total,
            )
        else:
            self.create_optimizer_and_scheduler(num_training_steps=t_total)
            optimizer = self.optimizer

        scheduler = self.lr_scheduler

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0

        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        optimizer.zero_grad(set_to_none=True)
        metrics = None
        progress_bar = tqdm(range(t_total))

        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):
                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                    
                if self.args.zero_order_optim and self.args.use_squeezellm:
                    tr_loss += self.set_squeezellm_sparse_grad(self.model, inputs)

                elif self.args.zero_order_optim and has_mask:
                    zo_random_seed = np.random.randint(1000000000)
                    tr_loss += self.set_grad_with_mask(self.model, inputs, self.outlier_masks, zo_random_seed)

                elif self.args.zero_order_optim:
                    tr_loss += self.zo_step(self.model, inputs)

                # standard, non-ZO optimization
                else:
                    tr_loss += self.training_step(model, inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16:
                        norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if self.args.optimizer_variant == 'signgd':
                        for n,p in model.named_parameters():
                            if p.grad is not None:
                                p.grad = torch.sign(p.grad)

                    if transformers.is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.state.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if has_mask:
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if not name in self.named_parameters_to_optim: continue
                                param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]


                    if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                        self.state.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["norm"] = norm.item()
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)
                        logger.info(str(logs))


                    if self.args.dynamic_mask and \
                            (self.state.global_step + 1) % self.args.dynamic_mask_step == 0:

                        if self.args.outlier:
                            self.outlier_masks = self.get_outlier_masks(self.model, self.args.outlier_percentage)

                        elif self.args.random_subset_weights:
                            self.outlier_masks = self.get_random_masks(self.model, self.args.outlier_percentage)

                        elif self.args.grad_mask:
                            self.outlier_masks = self.get_gradient_masks(self.model, self.args.outlier_percentage)

                        self.named_parameters_to_optim = {
                            n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in self.model.named_parameters() if n in self.outlier_masks
                        }
                        self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)

                progress_bar.update(1)
                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                # train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
    
    @torch.inference_mode()
    def squeezellm_zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, buffer in self.dequantized_weight_list:
            indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
            z = torch.normal(mean=0, std=1, size=(indices.shape[1],), device=buffer.device, dtype=buffer.dtype)
            buffer[indices[0], indices[1]] += (scaling_factor * self.args.zero_order_eps) * z
    
    @torch.inference_mode()
    def zo_perturb_parameters_with_mask(self, random_seed=None, mask_dict=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.model.named_parameters():
            if not name in self.named_parameters_to_optim: continue
            selected_param = self.named_parameters_to_optim[name]
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            param.view(-1)[mask_dict[name]] += (scaling_factor * self.args.zero_order_eps) * z


    @torch.inference_mode()
    def set_squeezellm_sparse_grad(self, model, inputs):
        self.zo_random_seed = np.random.randint(1000000000)

        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zero_order_eps)).item()

        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        
        torch.manual_seed(self.zo_random_seed)

        # with self.profiler('set_grad'):
        for name, buffer in self.dequantized_weight_list:
            sensitive_vals =  self.sqllm_sparse_weight_dict[name]
            z = torch.normal(mean=0, std=1, size=sensitive_vals.size(), device=sensitive_vals.device, dtype=sensitive_vals.dtype)
            sensitive_vals.grad = self.projected_grad * z

        return (loss1 + loss2) / 2

    def set_zo_grad_as_grad(self):
        """
        Update the parameters with the estimated gradients.
        """
        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     
        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.grad = self.projected_grad * z
