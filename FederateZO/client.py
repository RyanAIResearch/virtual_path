# -*- coding: UTF-8 -*-
#
# ProjectName: PromptZO-master
# Description: 
# Author: 

import pandas as pd
import numpy as np
import torch
import transformers
from transformers import default_data_collator, AutoModelForCausalLM, AutoModelForMaskedLM
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm
from ZO import *
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
import contextlib
import gc
from accelerate import Accelerator
import datasets
import math
import inspect
from .training_utils import (
    find_labels,
    seed_worker
)

class RemoveColumnsCollator:
    """Wrap the data collator to remove unused columns before they are passed to the collator."""

    def __init__(
        self,
        data_collator,
        signature_columns,
        logger=None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.data_collator = data_collator
        self.signature_columns = signature_columns
        self.logger = logger
        self.description = description
        self.model_name = model_name
        self.message_logged = False

    def _remove_columns(self, feature: dict) -> dict:
        if not isinstance(feature, dict):
            return feature
        if not self.message_logged and self.logger and self.model_name:
            ignored_columns = list(set(feature.keys()) - set(self.signature_columns))
            if len(ignored_columns) > 0:
                dset_description = "" if self.description is None else f"in the {self.description} set"
                self.logger.info(
                    f"The following columns {dset_description} don't have a corresponding argument in "
                    f"`{self.model_name}.forward` and have been ignored: {', '.join(ignored_columns)}."
                    f" If {', '.join(ignored_columns)} are not expected by `{self.model_name}.forward`, "
                    " you can safely ignore this message."
                )
                self.message_logged = True
        return {k: v for k, v in feature.items() if k in self.signature_columns}

    def __call__(self, features: List[dict]):
        features = [self._remove_columns(feature) for feature in features]
        return self.data_collator(features)


class Client(object):
    def __init__(self, args, local_train_data, client_model=None, seed_list=None, outlier_masks=None, named_parameters_to_optim=None, data_collator=None, id=-1):

        self.args = args

        self.local_model = client_model

        self.client_id = id

        self.train_dataset = local_train_data
        print("traindataset: ", len(self.train_dataset))

        self.received_seed_list = seed_list

        # Mask
        self.outlier_masks = outlier_masks

        self.named_parameters_to_optim = named_parameters_to_optim

        # current execution step in train_dataloader
        self.execute_step = 0

        # self.accelerator = Accelerator()

        self._signature_columns = None

        self.is_deepspeed_enabled = False
        
        self.data_collator = data_collator

        self.use_cpu_amp = False

        if self.args.training_mode == "loss_dynamic_step":
            self.step_counter = 0
            self.noniid_data_pointer = 0
        
        if self.args.training_mode == "sparse_dynamic_localstep":
            self.step_counter_sparse_dynamic = 0
            self.sparse_datapointer = 0
        
        if self.args.training_mode == "sparse_dynamic_multiclients" or self.args.training_mode == "sparse_dynamic_earlystppping":
            self.sparse_dynamic_multiclients_step_counter = 0
            self.sparse_dynamic_multiclients_datapointer = 0


        self.sampler = RandomSampler(self.train_dataset, replacement=True)

        if self.args.task_name == "wiki":

            self.train_dataloader = DataLoader(self.train_dataset, sampler=self.sampler, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
            
        else:

            self.train_dataloader = DataLoader(self.train_dataset, sampler=self.sampler, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)

            if self.args.training_mode == "loss_dynamic_step":
                self.data_loader_iter = iter(self.train_dataloader)

            self.total_step = len(self.train_dataloader)
    

    def receive_server_info(self, receive_seed_list=None, client_model=None, named_parameters_to_optim=None):
        if self.args.has_mask:
            self.received_seed_list = receive_seed_list
            if client_model is not None:
                self.local_model = client_model
            # default_label_names = find_labels(self.local_model.__class__)
            # self.label_names = default_label_names
            self.named_parameters_to_optim = named_parameters_to_optim
            with torch.no_grad():
                for name, param in self.local_model.named_parameters():
                    if not name in self.named_parameters_to_optim: continue
                    param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]
            if self.args.training_mode == "sparse_dynamic_multiclients" or self.args.training_mode == "sparse_dynamic_earlystppping":
                self.local_client_sparse_optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
                self.local_lr_sparse_scheduler = transformers.get_constant_schedule(self.local_client_sparse_optimizer)
            # self.train_dataloader = self.get_train_dataloader()
            # self.total_step = len(self.train_dataloader)
        else:
            self.received_seed_list = receive_seed_list
            self.local_model = client_model
            # if not hasattr(self, 'train_dataloader'):
            #     print("initial train dataloader")
            #     default_label_names = find_labels(self.local_model.__class__)
            #     self.label_names = default_label_names
            #     self.train_dataloader = self.get_train_dataloader()
            #     self.total_step = len(self.train_dataloader)


    # zo sparse mask forward
    @torch.inference_mode()
    def set_grad_with_mask(self, model, inputs, mask_dict, random_seed):

        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)

        loss1 = self.zo_forward(model, inputs)

        self.zo_perturb_parameters_with_mask(scaling_factor=-2, random_seed=random_seed, mask_dict=mask_dict)

        loss2 = self.zo_forward(model, inputs)

        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)

        global_projected_grad = ((loss1 - loss2) / (2 * self.args.ZO_eps)).item()
        del loss1
        del loss2
        gc.collect()
        torch.cuda.empty_cache()
        return global_projected_grad


    @torch.inference_mode()
    def zo_perturb_parameters_with_mask(self, random_seed=None, mask_dict=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.local_model.named_parameters():
            if not name in self.named_parameters_to_optim: continue
            selected_param = self.named_parameters_to_optim[name]
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            param.view(-1)[mask_dict[name]] += (scaling_factor * self.args.ZO_eps) * z
            del z

    
    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            # if self.args.n_gpu > 1:
            #     # Warning: this is copied from the original Huggingface Trainer. Untested.
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()
    

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        labels = None
        outputs = model(**inputs)

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
    

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": "cuda"}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )

        return inputs
    
    def compute_loss_context_manager(self):
        """
        A helper wrapper to group together context managers.
        """
        return self.autocast_smart_context_manager()

    def autocast_smart_context_manager(self, cache_enabled: Optional[bool] = True):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        if self.use_cpu_amp:
            ctx_manager = torch.cpu.amp.autocast(cache_enabled=cache_enabled, dtype=self.amp_dtype)
        else:
            ctx_manager = contextlib.nullcontext()

        return ctx_manager

    
    # client training
    def local_train(self, global_step=None):
        completed_steps = 0
        logging_interval = 10
        eval_steps = 50
        cur_loss = 0
        losses = []
        ppls = []

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch)
        
        # Train!
        print("***** Running training *****")
        if self.args.training_mode == "loss_single_batch":
            progress_bar = tqdm(range(len(self.received_seed_list)))
        elif self.args.training_mode == "weight_full_batch":
            progress_bar = tqdm(range(self.total_step))
        elif self.args.training_mode == "loss_full_batch":
            progress_bar = tqdm(range(self.total_step))
        elif self.args.training_mode == "loss_dynamic_step":
            progress_bar = tqdm(range(self.args.local_train_steps))
        elif self.args.training_mode == "sparse_dynamic_localstep":
            progress_bar = tqdm(range(self.args.local_train_steps))
        elif self.args.training_mode == "sparse_dynamic_multiclients" or self.args.training_mode == "sparse_dynamic_earlystppping":
            progress_bar = tqdm(range(self.args.local_train_steps))
        cur_loss = 0
        if self.args.training_mode == "sparse_dynamic_multiclients" or self.args.training_mode == "sparse_dynamic_earlystppping":
            project_grad_collector = {"client_id": self.client_id, "global_step": global_step, "steps": []}
        else:
            project_grad_collector = {"client_id" : self.client_id}

        if self.args.training_mode == "loss_single_batch":
            if self.args.has_mask:
                with torch.no_grad():
                    for step, batch in enumerate(self.train_dataloader):
                        if step < self.execute_step: continue
                        self.local_model.eval()
                        batch = {k: v.cuda() for k, v in batch.items()}
                        zo_sample = self.args.zo_sample
                        for seed in self.received_seed_list:
                            
                            projected_grad = self.set_grad_with_mask(self.local_model, batch, self.outlier_masks ,seed)

                            project_grad_collector[seed] = projected_grad

                            progress_bar.update(1)

                        self.execute_step += 1
                        del batch
                        if self.execute_step > self.total_step-1:
                            self.execute_step = 0
                        break
                    torch.cuda.empty_cache() 
                    return project_grad_collector
            else:
                with torch.no_grad():
                    for step, batch in enumerate(self.train_dataloader):
                        if step < self.execute_step: continue
                        self.local_model.eval()
                        batch = {k: v.cuda() for k, v in batch.items()}
                        zo_sample = self.args.zo_sample
                        for seed in self.received_seed_list:
                   
                            self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
                            # loss1 = self.local_model(**batch).loss
                            loss1 = self.zo_forward(self.local_model, batch)

                            self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=-2, seed=seed)                    
                            # loss2 = self.local_model(**batch).loss
                            loss2 = self.zo_forward(self.local_model, batch)

                            self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
                            cur_loss += (loss1 + loss2) / (2 * len(self.received_seed_list))
                            projected_grad = (loss1 - loss2) / (self.args.ZO_eps)

                            project_grad_collector[seed] = projected_grad

                            progress_bar.update(1)

                        self.execute_step += 1

                        if self.execute_step > self.total_step-1:
                            self.execute_step = 0
                        break
                    torch.cuda.empty_cache()
                    return project_grad_collector
                
        elif self.args.training_mode == "weight_full_batch":

            with torch.no_grad():
                for step, batch in enumerate(self.train_dataloader):
                    self.local_model.eval()
                    batch = {k: v.cuda() for k, v in batch.items()}
                    zo_sample = self.args.zo_sample
                    for _ in range(zo_sample):
                        seed = np.random.randint(10000000)
                        self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
                        loss1 = self.local_model(**batch).loss

                        self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=-2, seed=seed)                    
                        loss2 = self.local_model(**batch).loss

                        self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
                        cur_loss += (loss1 + loss2) / (2 * zo_sample)
                        projected_grad = (loss1 - loss2) / (self.args.ZO_eps)

                        torch.manual_seed(seed)

                        for n, p in self.local_model.named_parameters():
                            if n not in project_grad_collector:
                                project_grad_collector[n] = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(projected_grad/self.args.zo_sample)
                            else:
                                project_grad_collector[n].add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=projected_grad/self.args.zo_sample)
                    progress_bar.update(1)
                return project_grad_collector
        
        elif self.args.training_mode == "loss_full_batch":
            with torch.no_grad():
                for step, batch in enumerate(self.train_dataloader):
                    self.local_model.eval()
                    batch = {k: v.cuda() for k, v in batch.items()}
                    zo_sample = self.args.zo_sample
                    for seed in self.received_seed_list:
                        current_seed = seed
                        self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=current_seed)                    
                        loss1 = self.local_model(**batch).loss

                        self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=-2, seed=current_seed)                    
                        loss2 = self.local_model(**batch).loss

                        self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=current_seed)                    
                        cur_loss += (loss1 + loss2) / (2 * zo_sample)
                        projected_grad = (loss1 - loss2) / (self.args.ZO_eps)
                        # projected_grad = projected_grad
                        if seed not in project_grad_collector:
                            project_grad_collector[seed] = []
                            project_grad_collector[seed].append(projected_grad)
                        else:
                            project_grad_collector[seed].append(projected_grad)
                    progress_bar.update(1)
                for key, value in project_grad_collector.items():
                    if key == "client_id": continue
                    project_grad_collector[key] = sum(value) / len(value)
                return project_grad_collector
        
        elif self.args.training_mode == "sparse_dynamic_earlystppping":
            with torch.no_grad():
                while self.sparse_dynamic_multiclients_step_counter < self.args.local_train_steps:
                    for step, batch in enumerate(self.train_dataloader):
                        if step != self.sparse_dynamic_multiclients_datapointer:
                            continue
                        per_step_recorder = {}
                        self.local_model.eval()
                        batch = {k: v.cuda() for k, v in batch.items()}
                        zo_sample = self.args.zo_sample
                        target_seed_list = self.received_seed_list[self.sparse_dynamic_multiclients_step_counter]
                        # update the client named_parameters_to_optim
                        for seed in target_seed_list:
                            projected_grad = self.set_grad_with_mask(self.local_model, batch, self.outlier_masks ,seed)
                            per_step_recorder[seed] = projected_grad
                            torch.manual_seed(seed)
                            for name, selected_param in self.named_parameters_to_optim.items():
                                z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
                                if selected_param.grad is None:
                                    selected_param.grad = z * (projected_grad/self.args.zo_sample)
                                else:
                                    selected_param.grad.add_(z, alpha=(projected_grad/self.args.zo_sample))
                        project_grad_collector["steps"].append(per_step_recorder)
                        self.local_client_sparse_optimizer.step()
                        self.local_client_sparse_optimizer.zero_grad(set_to_none=True)
                        self.local_lr_sparse_scheduler.step()
                        # update the local model
                        with torch.no_grad():
                            # with self.profiler('optimizer'):
                            for name, param in self.local_model.named_parameters():
                                if not name in self.named_parameters_to_optim: continue
                                param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]

                        progress_bar.update(1)
                        print("====finish local training one step====")
                        # data pointer step
                        if self.sparse_dynamic_multiclients_datapointer == len(self.train_dataloader)-1:
                            self.sparse_dynamic_multiclients_datapointer = 0
                        else:
                            self.sparse_dynamic_multiclients_datapointer += 1
                        # control the local train steps
                        self.sparse_dynamic_multiclients_step_counter += 1
                        if self.sparse_dynamic_multiclients_step_counter == self.args.local_train_steps:
                            break
                self.sparse_dynamic_multiclients_step_counter = 0
                del self.local_client_sparse_optimizer
                del self.local_lr_sparse_scheduler
                gc.collect()
                torch.cuda.empty_cache()
                return project_grad_collector

        elif self.args.training_mode == "sparse_dynamic_multiclients":
            with torch.no_grad():
                while self.sparse_dynamic_multiclients_step_counter < self.args.local_train_steps:
                    for step, batch in enumerate(self.train_dataloader):
                        if step != self.sparse_dynamic_multiclients_datapointer:
                            continue
                        per_step_recorder = {}
                        self.local_model.eval()
                        batch = {k: v.cuda() for k, v in batch.items()}
                        zo_sample = self.args.zo_sample
                        target_seed_list = self.received_seed_list[self.sparse_dynamic_multiclients_step_counter]
                        # update the client named_parameters_to_optim
                        for seed in target_seed_list:
                            projected_grad = self.set_grad_with_mask(self.local_model, batch, self.outlier_masks ,seed)
                            per_step_recorder[seed] = projected_grad
                            torch.manual_seed(seed)
                            for name, selected_param in self.named_parameters_to_optim.items():
                                z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
                                if selected_param.grad is None:
                                    selected_param.grad = z * (projected_grad/self.args.zo_sample)
                                else:
                                    selected_param.grad.add_(z, alpha=(projected_grad/self.args.zo_sample))
                        project_grad_collector["steps"].append(per_step_recorder)
                        self.local_client_sparse_optimizer.step()
                        self.local_client_sparse_optimizer.zero_grad(set_to_none=True)
                        self.local_lr_sparse_scheduler.step()
                        # update the local model
                        with torch.no_grad():
                            # with self.profiler('optimizer'):
                            for name, param in self.local_model.named_parameters():
                                if not name in self.named_parameters_to_optim: continue
                                param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]

                        progress_bar.update(1)
                        print("====finish local training one step====")
                        # data pointer step
                        if self.sparse_dynamic_multiclients_datapointer == len(self.train_dataloader)-1:
                            self.sparse_dynamic_multiclients_datapointer = 0
                        else:
                            self.sparse_dynamic_multiclients_datapointer += 1
                        # control the local train steps
                        self.sparse_dynamic_multiclients_step_counter += 1
                        if self.sparse_dynamic_multiclients_step_counter == self.args.local_train_steps:
                            break
                self.sparse_dynamic_multiclients_step_counter = 0
                del self.local_client_sparse_optimizer
                del self.local_lr_sparse_scheduler
                gc.collect()
                torch.cuda.empty_cache()
                return project_grad_collector        

        
        elif self.args.training_mode == "sparse_dynamic_localstep":
            with torch.no_grad():
                while self.step_counter_sparse_dynamic < self.args.local_train_steps:
                    for step, batch in enumerate(self.train_dataloader):
                        if step != self.sparse_datapointer:
                            continue
                        self.local_model.eval()
                        batch = {k: v.cuda() for k, v in batch.items()}
                        zo_sample = self.args.zo_sample
                        target_seed_list = self.received_seed_list[self.step_counter_sparse_dynamic]
                        for seed in target_seed_list:
                            projected_grad = self.set_grad_with_mask(self.local_model, batch, self.outlier_masks ,seed)
                            project_grad_collector[seed] = projected_grad
                        progress_bar.update(1)

                        self.sparse_datapointer += 1
                        if self.sparse_datapointer == len(self.train_dataloader):
                            self.sparse_datapointer = 0
                        del batch
                        self.step_counter_sparse_dynamic += 1
                        if self.step_counter_sparse_dynamic == self.args.local_train_steps:
                            break
                self.step_counter_sparse_dynamic = 0
                torch.cuda.empty_cache()
                return project_grad_collector



        elif self.args.training_mode == "loss_dynamic_step":
            if self.args.iid:
                with torch.no_grad():
                    while self.step_counter < self.args.local_train_steps:
                        for step, batch in enumerate(self.train_dataloader):
                            self.local_model.eval()
                            batch = {k: v.cuda() for k, v in batch.items()}
                            zo_sample = self.args.zo_sample
                            target_seed_list = self.received_seed_list[self.step_counter]
                            for seed in target_seed_list:
                                self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
                                
                                loss1 = self.zo_forward(self.local_model, batch)
                                
                                self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=-2, seed=seed)                    
                                
                                loss2 = self.zo_forward(self.local_model, batch)
                                
                                self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
                                cur_loss += (loss1 + loss2) / (2 * len(self.received_seed_list))
                                projected_grad = (loss1 - loss2) / (self.args.ZO_eps)
                                
                                project_grad_collector[seed] = projected_grad
                                
                                progress_bar.update(1)
                            
                            self.step_counter += 1
                    torch.cuda.empty_cache()
                    self.step_counter = 0
                    return project_grad_collector
            else:
                with torch.no_grad():
                    while self.step_counter < self.args.local_train_steps:
                        for step, batch in enumerate(self.train_dataloader):
                            if self.noniid_data_pointer != step:
                                continue
                            self.local_model.eval()
                            batch = {k: v.cuda() for k, v in batch.items()}
                            zo_sample = self.args.zo_sample
                            target_seed_list = self.received_seed_list[self.step_counter]
                            for seed in target_seed_list:
                                self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
                                
                                loss1 = self.zo_forward(self.local_model, batch)
                                
                                self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=-2, seed=seed)                    
                                
                                loss2 = self.zo_forward(self.local_model, batch)
                                
                                self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
                                cur_loss += (loss1 + loss2) / (2 * len(self.received_seed_list))
                                projected_grad = (loss1 - loss2) / (self.args.ZO_eps)
                                
                                project_grad_collector[seed] = projected_grad
                                
                                progress_bar.update(1)

                            if self.noniid_data_pointer == (len(self.train_dataloader)-1):
                                self.noniid_data_pointer = 0
                            else:
                                self.noniid_data_pointer += 1
                            self.step_counter += 1
                            if self.step_counter == self.args.local_train_steps:
                                break
                    torch.cuda.empty_cache()
                    self.step_counter = 0
                    return project_grad_collector
                        

    # Data Processing
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(self.args, data_collator, None, self.local_model, self.label_names, description="training")
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": 0,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = False
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    
    def _get_collator_with_removed_columns(
        self, args, data_collator: Callable, _signature_columns, model, label_names, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = _signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            description=description,
            model_name=model.__class__.__name__,
        )
        return remove_columns_collator
    

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        return RandomSampler(self.train_dataset)


    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            print(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.local_model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.local_model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                "No columns in the dataset match the model's forward method signature. "
                f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "
                "Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`."
            )

        return dataset.remove_columns(ignored_columns)
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.local_model
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
        
    
            



            
            

            # with torch.no_grad():
            #     for step, batch in enumerate(self.train_dataloader):
            #         self.local_model.eval()
            #         batch = {k: v.cuda() for k, v in batch.items()}
            #         zo_sample = self.args.zo_sample
            #         # for _ in range(zo_sample):
            #         for seed in self.received_seed_list:
            #             # seed = np.random.randint(10000000)
            #             self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
            #             loss1 = self.local_model(**batch).loss

            #             self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=-2, seed=seed)                    
            #             loss2 = self.local_model(**batch).loss

            #             self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
            #             cur_loss += (loss1 + loss2) / (2 * len(self.received_seed_list))
            #             projected_grad = (loss1 - loss2) / (self.args.ZO_eps)
            #             # projected_grad = projected_grad
            #             if seed not in project_grad_collector:
            #                 project_grad_collector[seed] = []
            #             else:
            #                 project_grad_collector[seed].append(projected_grad)
            #         progress_bar.update(1)
            #     return project_grad_collector






            # with torch.no_grad():
            #     for step, batch in enumerate(self.train_dataloader):
            #         self.local_model.eval()
            #         batch = {k: v.cuda() for k, v in batch.items()}
            #         zo_sample = self.args.zo_sample
            #         for _ in range(zo_sample):
            #         # for seed in self.received_seed_list:
            #             seed = np.random.randint(10000000)
            #             self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
            #             loss1 = self.local_model(**batch).loss

            #             self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=-2, seed=seed)                    
            #             loss2 = self.local_model(**batch).loss

            #             self.local_model = perturb_parameters_MeZO(self.local_model, self.args.ZO_eps, scaling_factor=1, seed=seed)                    
            #             cur_loss += (loss1 + loss2) / (2 * len(self.received_seed_list))
            #             projected_grad = (loss1 - loss2) / (self.args.ZO_eps)
            #             # projected_grad = projected_grad
            #             project_grad_collector[seed] = projected_grad
            #         progress_bar.update(1)
            #     return project_grad_collector

                