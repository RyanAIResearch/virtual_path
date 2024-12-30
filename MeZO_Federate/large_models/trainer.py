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
from metrics import f1
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
import random
import transformers
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from metrics import calculate_metric

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    get_reporting_integration_callbacks,
    hp_params,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    WEIGHTS_NAME,
    find_labels,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)
from transformers.utils.generic import ContextManagers
import wandb
from utils import encode_prompt, Prediction
from FlZO.server import Server
from FlZO.client import Client

_is_native_cpu_amp_available = True

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_datasets_available():
    import datasets


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def get_modules(self, layer, model_type):
        # NOTE: This is llama-specific
        # For other models, replace this with proper names for all linear layers
        if model_type in ['mistral', 'llama']:
            return[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ]
        elif model_type in ['opt']:
            return[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.out_proj,
                layer.fc1,
                layer.fc2,
            ]
        else:
            raise NotImplementedError()


    def get_grad_and_load_to_model(self, model: nn.Module, grad_save_path,
                                   microbatch=1, minibatch=1000):
        count = 0
        self._train_batch_size = microbatch

        _model = model.model
        if 'mistral' in self.args.model_name.lower():
            _layers = _model.layers
            model_type = 'mistral'

        elif 'llama' in self.args.model_name.lower():
            _layers = _model.layers
            model_type = 'llama'

        elif 'opt' in self.args.model_name.lower():       
            _layers = _model.decoder.layers
            model_type = 'opt'

        else:
            raise NotImplementedError()

        for n, p in model.named_parameters():
            p.requires_grad_(False)
        
        for layer in _layers:
            for module in self.get_modules(layer, model_type):
                module.weight.requires_grad_(True)
        
        minibatch = min(minibatch, len(self.get_train_dataloader()))
        counter = tqdm(range(minibatch))
        continue_count = 0
        with torch.enable_grad():
            for sampled_batch in self.get_train_dataloader():
                sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}    
                if sampled_batch['input_ids'].shape[0] * sampled_batch['input_ids'].shape[1] > 2700:
                    continue_count += 1
                    print(continue_count)
                    continue
                outputs = model(**sampled_batch)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    loss = outputs[0]
                loss.backward()
                torch.cuda.empty_cache()
                count += 1
                counter.update(1)
                
                if count >= minibatch: 
                    break

        _grad_module_dict = { (i, j): torch.zeros_like(module.weight, device='cpu')
            for i, layer in enumerate(_layers) 
            for j, module in enumerate(self.get_modules(layer, model_type)) 
        }
        for i, layer in enumerate(_layers):
            for j, module in enumerate(self.get_modules(layer, model_type)):
                _grad_module_dict[(i, j)] = module.weight.grad.data / count
        model.zero_grad(set_to_none=True)
        
        torch.save(_grad_module_dict, grad_save_path)   


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
    def get_gradient_masks(self, model: nn.Module,
                                    percentile=5e-3, microbatch=1, minibatch=16):
        assert minibatch % microbatch == 0
        count = 0
        old_batch_size = self._train_batch_size
        self._train_batch_size = microbatch
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

        self._train_batch_size = old_batch_size
        return mask_dict

    @torch.no_grad()
    def get_C4_gradient_masks(self, model, C4_grad_addr, percentile):
        mask_dict = dict()
        grad = transformers.AutoModelForCausalLM.from_pretrained(
            C4_grad_addr,
        )
        grad_dict = {n : p for n, p in grad.named_parameters()}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
                per_layer_scores = grad_dict[n].cuda() ** 2
                top_cutoff = int(p.numel() * percentile)
                mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
                mask[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
                mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]

        return mask_dict


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

    @torch.no_grad()
    def add_to_weight(self, model, vec, scale):
        counter = 0
        for p in model.parameters():
            if p.requires_grad:
                p.data.add_(vec[counter : counter + p.numel()].view(p.shape), alpha=scale)
                counter += p.numel()

    def get_grad(self, model, inputs, random_seed):
        self.zo_perturb_parameters(scaling_factor=1, random_seed=random_seed)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=-2, random_seed=random_seed)
        loss2 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=1, random_seed=random_seed)
        global_projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        torch.manual_seed(random_seed)
        ret_grad_dict = dict()
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            ret_grad_dict[name] = global_projected_grad * z
        return ret_grad_dict, loss1

    def get_grad_rademacher(self, model, inputs, random_seed):
        self.zo_perturb_parameters_rademacher(scaling_factor=1, random_seed=random_seed)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters_rademacher(scaling_factor=-2, random_seed=random_seed)
        loss2 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters_rademacher(scaling_factor=1, random_seed=random_seed)
        global_projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        torch.manual_seed(random_seed)
        ret_grad_dict = dict()
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            # Resample z
            z = torch.randint(low=0, high=2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            z[z == 0] = -1
            ret_grad_dict[name] = global_projected_grad * z
        return ret_grad_dict, loss1



    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # with self.profiler('forward-backward'):
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            with self.profiler('forward-backward'):
                loss.backward()

        return loss.detach()

    @torch.no_grad()
    def perturb_parameters_one_column_LoRA(self,
                                           model: nn.Module, 
                                       column_idx, 
                                       column_full_param_pair_list, 
                                       need_perturb=False,
                                       random_vectors=None, 
                                       eps=-1,
                                       scaling_factor=-1):
        if random_vectors is None:
            random_vectors = dict()
        for n, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            else:
                if 'lora_A' in n:
                    col, mat = column_full_param_pair_list[n]
                    if need_perturb:
                        if n not in random_vectors:
                            random_vectors[n] = torch.randn_like(col)
                        p[column_idx, :].add_(random_vectors[n], alpha=scaling_factor*eps)
                    else:
                        p.data.copy_(mat)
                else:
                    col, mat = column_full_param_pair_list[n]
                    if need_perturb:
                        if n not in random_vectors:
                            random_vectors[n] = torch.randn_like(col)
                        p[:, column_idx].add_(random_vectors[n], alpha=scaling_factor*eps)
                    else:
                        p.data.copy_(mat)
        return model, random_vectors

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def evaluate_test_set(self):
        # Prediction loop
        predictions = []  
        train_samples = self.train_samples
        for eval_id, eval_sample in enumerate(tqdm(self.test_samples)):
            predictions.append(
                self.one_step_pred(train_samples, eval_sample)
            )
        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        print(metrics, flush=True)
        sys.stdout.flush()
        return metrics

    def one_step_pred(self, train_samples, eval_sample):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """

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
            output_text = self.test_forward(encoded_candidates[0], generation=True)
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.test_forward(encoded_candidate, option_len=option_lens[candidate_id])

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.test_forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))
    
    def split_data(self, train_data):
        random.shuffle(train_data)

        num_samples_per_client = len(train_data) // self.args.total_num_users

        client_datasets = []
        for i in range(self.args.total_num_users):
            start = i * num_samples_per_client
            end = start + num_samples_per_client if i < self.args.total_num_users - 1 else len(train_data)
            client_datasets.append(train_data[start:end])
        return client_datasets


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        self.test_dataset = self.eval_dataset['test']

        test_dataloader = self.get_test_dataloader(self.test_dataset)

        client_data_lists = self.split_data(self.train_dataset.__get__())

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if self.args.save_grad:
            self.get_grad_and_load_to_model(self.model, self.args.grad_save_path + os.path.sep + f'start.pt')
            raise RuntimeError('finished')
        delay_optimizer_creation = False
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        has_mask = False
        if self.args.smaller_weight_mask:
            print(f'smaller weight mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_smaller_weight_masks(model, self.args.outlier_percentage)
            has_mask = True
            
        elif self.args.outlier:
            print(f'outlier mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_outlier_masks(model, self.args.outlier_percentage)
            has_mask = True

        elif self.args.random_subset_weights:
            print(f'random mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_random_masks(model, self.args.outlier_percentage)
            has_mask = True

        elif self.args.grad_mask:
            print(f'grad mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_gradient_masks(model, self.args.outlier_percentage)
            has_mask = True

        elif self.args.C4_grad_mask:
            print(f'C4 grad mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_C4_gradient_masks(model, self.args.C4_grad_addr, self.args.outlier_percentage)
            # self.outlier_masks = torch.load('C4_grad_mask_1e_3.pt')
            has_mask = True

        elif self.args.GraSP_mask:
            print(f'GraSP mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_GraSP_mask(model, self.args.outlier_percentage)
            has_mask = True

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        self.named_parameters_to_optim = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # FL Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        # 开始联邦学习训练
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")
        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad(set_to_none=True)

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        # if args.trainer != 'zo':
        if args.report_to != []:
            wandb.run.name = self.args.tag

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                _ = list(train_dataloader.sampler)

        from eventProfiler import EventProfiler
        # self.profiler = EventProfiler(torch.device('cuda'), active=False)
        self.record_loss = []

        if has_mask:
            self.named_parameters_to_optim = {
                n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in model.named_parameters() if n in self.outlier_masks
            }

        if args.use_adam:
            if has_mask:
                self.optimizer = torch.optim.Adam(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
            else:
                # with self.profiler('optimizer'):
                self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        
        elif args.use_momentum:
            if has_mask:
                self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate, momentum=0.9)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9)

        else:
            if has_mask:
                self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate)

        server = Server(self.args, train_dataloader, test_dataloader, model, self.named_parameters_to_optim, self.outlier_masks, self.optimizer, self.lr_scheduler)

        if args.training_mode == "loss_single_batch":
            clients = []
            # Initial Client Local Data
            for client_idx in range(args.total_num_users):
                local_train_data = client_data_lists[client_idx]
                client = Client(
                    args, local_train_data, 
                    self.tokenizer, self.data_collator, has_mask, resume_from_checkpoint, self.named_parameters_to_optim, 
                    self.outlier_masks, epochs_trained, None, None, client_idx
                )
                clients.append(client)

            # Start Training
            for e in range(num_train_epochs):

                print(f"Global epoch {e}...")
                client_projected_grad_accumulator = []
                m = max(int(args.frac * args.total_num_users), 1)
                #sample users for training
                idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
                for idx in idxs_users:
                    client_model = copy.deepcopy(model)
                    receive_seed_list = server.get_seed_list()
                    client_started = clients[idx]
                    client_started.receive_server_info(receive_seed_list, client_model)
                    grad_client = client_started.local_train()
                    client_projected_grad_accumulator.append(grad_client)
                    print(f"Clinet {idx} finish local training")
                    del client_model
                    torch.cuda.empty_cache() 

                model = server.model_aggregate(client_projected_grad_accumulator)
                del client_projected_grad_accumulator
                # ppl = server.model_eval(self.model)
                torch.cuda.empty_cache() 
                # print(f'Current PPL: {ppl:.3f}')
        elif args.training_mode == "loss_full_batch":
            # Start Training
            for e in range(num_train_epochs):
                print(f"Global epoch {e}...")
                client_projected_grad_accumulator = []
                m = max(int(self.args.frac * args.total_num_users), 1)
                #sample users for training
                idxs_users = np.random.choice(range(args.total_num_users), m, replace=False)
                for idx in idxs_users:
                    client_model = copy.deepcopy(model)
                    client_named_parameters_to_optim = copy.deepcopy(self.named_parameters_to_optim)
                    client_outlier_masks = copy.deepcopy(self.outlier_masks)
                    local_train_data = client_data_lists[idx]
                    client = Client(
                        args, local_train_data, 
                        self.tokenizer, self.data_collator, 
                        has_mask, resume_from_checkpoint, client_named_parameters_to_optim, 
                        client_outlier_masks, epochs_trained, client_model, None, idx
                    )
                    grad_client=client.local_train()
                    client_projected_grad_accumulator.append(grad_client)
                    print(f"Clinet {idx} finish local training")
                    del client
                    del client_model
                    del client_outlier_masks
                    del client_named_parameters_to_optim
                    torch.cuda.empty_cache() 
                model, self.named_parameters_to_optim, self.outlier_masks = server.model_aggregate(client_projected_grad_accumulator)
                del client_projected_grad_accumulator
                torch.cuda.empty_cache()
                ppl = server.model_eval(model)
                print(f'Current PPL: {ppl:.3f}')

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")      
        self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        # if self.args.record_time:
        #     torch.save(self.profiler.summary(), f'time{os.sep}{self.args.model_name.replace(os.sep, "-")}-{self.args.task_name}-{self.args.tag.replace(os.sep, "-")}-time.pt')
        #     torch.save(self.record_loss, f'time{os.sep}{self.args.model_name.replace(os.sep, "-")}-{self.args.task_name}-{self.args.tag.replace(os.sep, "-")}-loss.pt')
        return TrainOutput(self.state.global_step, train_loss, metrics)





    

    def zo_perturb_parameters_rademacher(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.randint(low=0, high=2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            z[z == 0] = -1
            param.data.add_(z, alpha=scaling_factor * self.args.zo_eps)


    def norm_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, normalize_dict=None):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps * normalize_dict[name]


    




    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")