# -*- coding: UTF-8 -*-
#
# ProjectName: PromptZO-master
# Description: 
# Author: 

import pandas as pd
import numpy as np
import tqdm
import torch, math
from transformers import default_data_collator
import transformers
import json
from .metrics import calculate_metric
import torch.nn.functional as F
from .datautils import get_loaders
from .training_utils import *
import gc
from ZO import *
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
import contextlib
import torch.nn.functional as F

class Server(object):

    def __init__(self, args, train_dataloader, eval_loader, global_model, outlier_masks, named_parameters_to_optim, c4_pretrain_grad=None, sparse_optimizer=None, task=None, tokenizer=None):

        self.args = args
        
        self.global_model = global_model

        self.eval_loader = eval_loader

        self.train_dataloader = train_dataloader

        self.outlier_masks = outlier_masks

        self.task = task

        self.named_parameters_to_optim = named_parameters_to_optim

        self.tokenizer = tokenizer

        if self.args.has_mask:
            self.optimizer = sparse_optimizer
        else:
            if self.args.adam:
                self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.args.learning_rate, 
                                            betas=(self.args.adam_beta1, self.args.adam_beta2), weight_decay=self.args.weight_decay)
            elif self.args.momentum:
                self.optimizer = torch.optim.SGD(self.global_model.parameters(), 
                                            lr=self.args.learning_rate, momentum=self.args.momentum_mu, weight_decay=self.args.weight_decay)
            elif self.args.SGD:
                self.optimizer = torch.optim.SGD(
                    self.global_model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            else:
                raise NotImplementedError()

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        if self.args.linear_lr:
            self.lr_scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 0, self.args.max_train_steps)

        elif self.args.cosine_lr:
            self.lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer, 0, self.args.max_train_steps)

        elif self.args.cosine_lr_with_warmup:
            self.lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer, int(0.1 * self.args.max_train_steps), self.args.max_train_steps)

        else:
            self.lr_scheduler = transformers.get_constant_schedule(self.optimizer)
        
        if self.args.training_mode == "sparse_dynamic_localstep" or self.args.training_mode == "sparse_dynamic_multiclients" or self.args.training_mode == "sparse_dynamic_earlystppping":
            self.path_recorder = {}
            self.cos_similarity_log = []
            self.norm_log = []
            self.dot_product = []
            self.server_check_seed_list = []

            self.central_server_checker = c4_pretrain_grad
            self.central_server_checker = self.central_server_checker / 10000
            self.central_server_checker = torch.sqrt(self.central_server_checker)
    
    def get_seed_list(self):
        # torch.manual_seed(3407)
        if self.args.training_mode == "loss_dynamic_step" or self.args.training_mode == "sparse_dynamic_localstep" or self.args.training_mode == "sparse_dynamic_multiclients" or self.args.training_mode == "sparse_dynamic_earlystppping":
            total_seed_list = []
            for i in range(self.args.local_train_steps):
                self.seed_lists = [np.random.randint(10000000) for _ in range(self.args.zo_sample)]
                total_seed_list.append(self.seed_lists)
            self.server_check_seed_list = total_seed_list
            return total_seed_list
        else:
            self.seed_lists = [np.random.randint(10000000) for _ in range(self.args.zo_sample)]
            print("seed_list: ", self.seed_lists)
            return self.seed_lists
    

    def model_aggregate(self, grad_accumulator, step=None):
        with torch.no_grad():
            self.global_model.eval()
            if self.args.training_mode == "loss_single_batch":
                if self.args.has_mask:
                    project_grad_accelerator = {}
                    # fill the dict
                    for per_client_proj_grad in grad_accumulator:
                        for key, value in per_client_proj_grad.items():
                            if key == 'client_id': continue
                            if key not in project_grad_accelerator:
                                project_grad_accelerator[key] = value
                            else:
                                project_grad_accelerator[key] += value
                    # average
                    for k, v in project_grad_accelerator.items():
                        project_grad_accelerator[k] /= len(grad_accumulator)
                    # optimize the chose 0.1% parameters in named_parameters_to_optim
                    for seed, project_grad in project_grad_accelerator.items():
                        torch.manual_seed(seed)
                        for name, selected_param in self.named_parameters_to_optim.items():
                            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
                            if selected_param.grad is None:
                                selected_param.grad = z * (project_grad/self.args.zo_sample)
                            else:
                                selected_param.grad.add_(z, alpha=(project_grad/self.args.zo_sample))
                   
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    with torch.no_grad():
                        # with self.profiler('optimizer'):
                        for name, param in self.global_model.named_parameters():
                            if not name in self.named_parameters_to_optim: continue
                            param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]

                    del project_grad_accelerator
                    del grad_accumulator
                    return self.global_model, self.named_parameters_to_optim
                else:
                    project_grad_accelerator = {}
                    # fill the dict
                    for per_client_proj_grad in grad_accumulator:
                        for key, value in per_client_proj_grad.items():
                            if key == 'client_id': continue
                            if key not in project_grad_accelerator:
                                project_grad_accelerator[key] = value
                            else:
                                project_grad_accelerator[key] += value
                    # average
                    for k, v in project_grad_accelerator.items():
                        project_grad_accelerator[k] /= len(grad_accumulator)
                    for seed, project_grad in project_grad_accelerator.items():
                        torch.manual_seed(seed)
                        for n, p in self.global_model.named_parameters():
                            if not p.requires_grad: continue
                            if p.grad is None:
                                p.grad = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(project_grad/self.args.zo_sample)
                            else:
                                p.grad.add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=project_grad/self.args.zo_sample)

                    del project_grad_accelerator
                    del grad_accumulator
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()
                    return self.global_model
            elif self.args.training_mode == "weight_full_batch":
                for client_proj_grad in grad_accumulator:
                    for n, p in self.global_model.named_parameters():
                        if not p.requires_grad: continue
                        if p.grad is None:
                            p.grad = client_proj_grad[n]/len(grad_accumulator)
                        else:
                            p.grad.add_(client_proj_grad[n]/len(grad_accumulator))
                del grad_accumulator
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                return self.global_model
            
            elif self.args.training_mode == "loss_full_batch":
                project_grad_collator = {}
                # collect each seed proj_grad
                for per_client_proj_grad in grad_accumulator:
                    for key, value in per_client_proj_grad.items():
                        if key == "client_id": continue
                        if key not in project_grad_collator:
                            project_grad_collator[key] = value
                        else:
                            project_grad_collator[key] += value
                # average:
                for cur_seed, project_grad in project_grad_collator.items():
                    project_grad_collator[cur_seed] = project_grad / len(grad_accumulator)
                # update the model
                for cur_key, proj_grad_avg in project_grad_collator.items():
                    torch.manual_seed(cur_key)
                    for n, p in self.global_model.named_parameters():
                        if not p.requires_grad: continue
                        if p.grad is None:
                            p.grad = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(proj_grad_avg/self.args.zo_sample)
                        else:
                            p.grad.add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=proj_grad_avg/self.args.zo_sample)
                del grad_accumulator
                del project_grad_collator
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                return self.global_model
            
            elif self.args.training_mode == "sparse_dynamic_earlystppping":
                project_grad_accelerator = {}
                # fill the dict
                print("calculate each client inner product with c4 grad")
                virtual_path = self.calculate_each_client_innerproduct(step, grad_accumulator)
                optimzed_virtual_path = self.early_stopping_checker(virtual_path)
                optimized_selected_path = self.selected_path(grad_accumulator, optimzed_virtual_path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                client_optim_parameters_dict = {}
                client_model_dict = {}
                for client_training_trace in optimized_selected_path:
                    cid = client_training_trace['client_id']
                    client_optim_parameters_dict[f"client_optim_parameter_{cid}"] = copy.deepcopy(self.named_parameters_to_optim)
                    client_model_dict[f"client_model_{cid}"] = copy.deepcopy(self.global_model)
                    server_monitor_client_optim_optimizer = torch.optim.SGD(list(client_optim_parameters_dict[f"client_optim_parameter_{cid}"].values()), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
                    server_monitor_client_optim_scheduler = transformers.get_constant_schedule(server_monitor_client_optim_optimizer)
                    for step_info in client_training_trace["steps"]:
                        for seed, project_grad in step_info.items():
                            torch.manual_seed(seed)
                            for n, p in client_optim_parameters_dict[f"client_optim_parameter_{cid}"].items():
                                if not p.requires_grad: continue
                                if p.grad is None:
                                    p.grad = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(project_grad/self.args.zo_sample)
                                else:
                                    p.grad.add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=project_grad/self.args.zo_sample)
                        server_monitor_client_optim_optimizer.step()
                        server_monitor_client_optim_optimizer.zero_grad(set_to_none=True)
                        server_monitor_client_optim_scheduler.step()

                        with torch.no_grad():
                            # with self.profiler('optimizer'):
                            for name, param in client_model_dict[f"client_model_{cid}"].named_parameters():
                                if not name in client_optim_parameters_dict[f"client_optim_parameter_{cid}"]: continue
                                param.view(-1)[self.outlier_masks[name]] = client_optim_parameters_dict[f"client_optim_parameter_{cid}"][name]
                    # model and client optim parameters to cpu
                    client_model_dict[f"client_model_{cid}"] = client_model_dict[f"client_model_{cid}"].to("cpu")
                    for name in client_optim_parameters_dict[f"client_optim_parameter_{cid}"].keys():
                        client_optim_parameters_dict[f"client_optim_parameter_{cid}"][name] = client_optim_parameters_dict[f"client_optim_parameter_{cid}"][name].to("cpu")
                    del server_monitor_client_optim_optimizer
                    del server_monitor_client_optim_scheduler
                    gc.collect()

                self.global_model = self.global_model.to("cpu")
                for p_global in self.global_model.parameters():
                    p_global.data.zero_()
                for i in range(self.args.total_num_users):
                    for p_global, p_client in zip(self.global_model.parameters(),
                                                client_model_dict[f"client_model_{i}"].parameters()):
                        p_global.data.add_(p_client.data)
                for p_global in self.global_model.parameters():
                    p_global.data.div_(self.args.total_num_users)
                self.global_model = self.global_model.to(device)
                del client_model_dict
                del client_optim_parameters_dict
                del grad_accumulator
                gc.collect()
                self.named_parameters_to_optim = {
                    n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in self.global_model.named_parameters() if n in self.outlier_masks
                }
                return self.global_model, self.named_parameters_to_optim
            
            elif self.args.training_mode == "sparse_dynamic_multiclients":
                project_grad_accelerator = {}
                # fill the dict
                print("calculate each client inner product with c4 grad")
                self.calculate_each_client_innerproduct(step, grad_accumulator)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                client_optim_parameters_dict = {}
                client_model_dict = {}
                for idx, client_training_trace in enumerate(grad_accumulator):
                    client_optim_parameters_dict[f"client_optim_parameter_{idx}"] = copy.deepcopy(self.named_parameters_to_optim)
                    client_model_dict[f"client_model_{idx}"] = copy.deepcopy(self.global_model)
                    server_monitor_client_optim_optimizer = torch.optim.SGD(list(client_optim_parameters_dict[f"client_optim_parameter_{idx}"].values()), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
                    server_monitor_client_optim_scheduler = transformers.get_constant_schedule(server_monitor_client_optim_optimizer)
                    for step_info in client_training_trace["steps"]:
                        for seed, project_grad in step_info.items():
                            torch.manual_seed(seed)
                            for n, p in client_optim_parameters_dict[f"client_optim_parameter_{idx}"].items():
                                if not p.requires_grad: continue
                                if p.grad is None:
                                    p.grad = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(project_grad/self.args.zo_sample)
                                else:
                                    p.grad.add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=project_grad/self.args.zo_sample)
                        server_monitor_client_optim_optimizer.step()
                        server_monitor_client_optim_optimizer.zero_grad(set_to_none=True)
                        server_monitor_client_optim_scheduler.step()

                        with torch.no_grad():
                            # with self.profiler('optimizer'):
                            for name, param in client_model_dict[f"client_model_{idx}"].named_parameters():
                                if not name in client_optim_parameters_dict[f"client_optim_parameter_{idx}"]: continue
                                param.view(-1)[self.outlier_masks[name]] = client_optim_parameters_dict[f"client_optim_parameter_{idx}"][name]
                    # model and client optim parameters to cpu
                    client_model_dict[f"client_model_{idx}"] = client_model_dict[f"client_model_{idx}"].to("cpu")
                    for name in client_optim_parameters_dict[f"client_optim_parameter_{idx}"].keys():
                        client_optim_parameters_dict[f"client_optim_parameter_{idx}"][name] = client_optim_parameters_dict[f"client_optim_parameter_{idx}"][name].to("cpu")
                    del server_monitor_client_optim_optimizer
                    del server_monitor_client_optim_scheduler
                    gc.collect()

                self.global_model = self.global_model.to("cpu")
                for p_global in self.global_model.parameters():
                    p_global.data.zero_()
                for i in range(self.args.total_num_users):
                    for p_global, p_client in zip(self.global_model.parameters(),
                                                client_model_dict[f"client_model_{i}"].parameters()):
                        p_global.data.add_(p_client.data)
                for p_global in self.global_model.parameters():
                    p_global.data.div_(self.args.total_num_users)
                self.global_model = self.global_model.to(device)
                del client_model_dict
                del client_optim_parameters_dict
                del grad_accumulator
                gc.collect()
                self.named_parameters_to_optim = {
                    n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in self.global_model.named_parameters() if n in self.outlier_masks
                }
                return self.global_model, self.named_parameters_to_optim



            elif self.args.training_mode == "sparse_dynamic_localstep":
                project_grad_accelerator = {}
                # fill the dict
                print("calculate each client inner product with c4 grad")
                self.calculate_each_client_innerproduct(step, grad_accumulator)
                for per_client_proj_grad in grad_accumulator:
                    for key, value in per_client_proj_grad.items():
                        if key == 'client_id': continue
                        if key not in project_grad_accelerator:
                            project_grad_accelerator[key] = value
                        else:
                            project_grad_accelerator[key] += value
                                            
                # average
                for k, v in project_grad_accelerator.items():
                    project_grad_accelerator[k] /= len(grad_accumulator)

                # optimize the chose 0.1% parameters in named_parameters_to_optim
                for seed, project_grad in project_grad_accelerator.items():
                    torch.manual_seed(seed)
                    for name, selected_param in self.named_parameters_to_optim.items():
                        z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
                        if selected_param.grad is None:
                            selected_param.grad = z * (project_grad/self.args.zo_sample)
                        else:
                            selected_param.grad.add_(z, alpha=(project_grad/self.args.zo_sample))

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()

                with torch.no_grad():
                    # with self.profiler('optimizer'):
                    for name, param in self.global_model.named_parameters():
                        if not name in self.named_parameters_to_optim: continue
                        param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]

                del project_grad_accelerator
                del grad_accumulator
                torch.cuda.empty_cache()
                return self.global_model, self.named_parameters_to_optim

            elif self.args.training_mode == "loss_dynamic_step":
                project_grad_accelerator = {}
                # fill the dict
                for per_client_proj_grad in grad_accumulator:
                    for key, value in per_client_proj_grad.items():
                        if key == 'client_id': continue
                        if key not in project_grad_accelerator:
                            project_grad_accelerator[key] = value
                        else:
                            project_grad_accelerator[key] += value
                # average
                for k, v in project_grad_accelerator.items():
                    project_grad_accelerator[k] /= len(grad_accumulator)
                for seed, project_grad in project_grad_accelerator.items():
                    torch.manual_seed(seed)
                    for n, p in self.global_model.named_parameters():
                        if not p.requires_grad: continue
                        if p.grad is None:
                            p.grad = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(project_grad/self.args.zo_sample)
                        else:
                            p.grad.add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=project_grad/self.args.zo_sample)
                
                del project_grad_accelerator
                del grad_accumulator
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                return self.global_model
    
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
        
        for name, param in self.global_model.named_parameters():
            if not name in self.named_parameters_to_optim: continue
            selected_param = self.named_parameters_to_optim[name]
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            param.view(-1)[mask_dict[name]] += (scaling_factor * self.args.ZO_eps) * z
            del z

    
    # def zo_forward(self, model, inputs):
    #     """
    #     Get (no gradient) loss from the model. Dropout is turned off too.
    #     """
    #     model.eval()

    #     with torch.inference_mode():
    #         inputs = self._prepare_inputs(inputs)
    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, inputs)
    #         # if self.args.n_gpu > 1:
    #         #     # Warning: this is copied from the original Huggingface Trainer. Untested.
    #         #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #     return loss.detach()

    def zo_forward(self, model, inputs):
        model.eval()
        with torch.inference_mode():
            outputs = model(input_ids=inputs, labels=inputs)
            loss = outputs.loss
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

    def early_stopping_checker(self, clients_innerproduct_list):
        remaining_clients = {client[0]["current_client_id"]: idx for idx, client in enumerate(clients_innerproduct_list)}
        max_steps = max(len(client) for client in clients_innerproduct_list)  # Determine maximum number of steps
        print("max_steps: ", max_steps)
        # Always keep the first step for all clients
        truncated_data = [[] for _ in clients_innerproduct_list]

        for idx, client in enumerate(clients_innerproduct_list):
            first_step = [entry for entry in client if entry["step"] == 1]
            if first_step:
                truncated_data[idx].extend(first_step)

        for step in range(2, max_steps + 1):  # Start from step 2 (exclude step 1)
            step_dot_products = []

            # Collect dot products for current step from remaining clients
            for client_id, idx in remaining_clients.items():
                client_steps = clients_innerproduct_list[idx]
                step_data = next((entry for entry in client_steps if entry["step"] == step), None)
                if step_data:
                    step_dot_products.append((step_data["dot_product"], client_id))
            # If no clients have data for this step, move to the next step
            if not step_dot_products:
                continue

            # Find the client with the smallest dot product at the current step
            min_dot_product, min_client_id = min(step_dot_products, key=lambda x: x[0])
            print("min_client_id: ", min_client_id)
            # Eliminate the client with the smallest dot product
            remaining_clients.pop(min_client_id, None)

            # If only one client remains, terminate early
            if len(remaining_clients) == 1:
                recorder_step = step
                for client_id, idx in remaining_clients.items():
                    while recorder_step <= max_steps:
                        step_data = next((entry for entry in clients_innerproduct_list[idx] if entry["step"] == recorder_step), None)
                        if step_data:
                            truncated_data[idx].append(step_data)
                        if recorder_step == max_steps:
                            break
                        recorder_step += 1
                break

            # Add the current step data to the truncated list for remaining clients
            for client_id, idx in remaining_clients.items():
                step_data = next((entry for entry in clients_innerproduct_list[idx] if entry["step"] == step), None)
                if step_data:
                    truncated_data[idx].append(step_data)
            print("remaining_clients: ", remaining_clients)
        return truncated_data
    
    def selected_path(self, grad_accumulator, optimized_virtual_path):
        client2steps = {}
        for item in grad_accumulator:
            cid = item['client_id']
            client2steps[cid] = item['steps']  # list of dicts
        # optim path result
        filtered_result = []

        for sublist in optimized_virtual_path:
            if not sublist:  # 如果子列表为空，就跳过
                continue
            cid = sublist[0]['current_client_id']
            needed_steps = len(sublist)
            all_steps = client2steps[cid]
            picked_steps = all_steps[:needed_steps]
            c_global_step = sublist[0]['current_global_step']

            filtered_result.append({
                'client_id': cid,
                'global_step': c_global_step,   # 这里示例都用 0，真实场景如需保留原 global_step，可自行修改
                'steps': picked_steps
            })

        return filtered_result



    def calculate_each_client_innerproduct(self, current_global_step, each_client_project_grad_list, initial_step=0):
        if self.args.training_mode == "sparse_dynamic_multiclients":
            for idx, client_training_trace in enumerate(each_client_project_grad_list):
                self.dot_product = []
                current_client_id = client_training_trace["client_id"]
                client_optim_parameters_dict = {}
                client_model_dict = {}
                client_optim_parameters_dict[f"client_optim_parameter_{idx}"] = copy.deepcopy(self.named_parameters_to_optim)
                client_model_dict[f"client_model_{idx}"] = copy.deepcopy(self.global_model)
                server_monitor_client_optim_optimizer = torch.optim.SGD(list(client_optim_parameters_dict[f"client_optim_parameter_{idx}"].values()), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
                server_monitor_client_optim_scheduler = transformers.get_constant_schedule(server_monitor_client_optim_optimizer)
                current_local_step_counter = 0
                for step_info in client_training_trace["steps"]:
                    for seed, project_grad in step_info.items():
                        torch.manual_seed(seed)
                        for n, p in client_optim_parameters_dict[f"client_optim_parameter_{idx}"].items():
                            if not p.requires_grad: continue
                            if p.grad is None:
                                p.grad = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(project_grad/self.args.zo_sample)
                            else:
                                p.grad.add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=project_grad/self.args.zo_sample)
                    current_local_step_counter += 1
                    selected_model_params_grad = self.flatten_parameter_gradients(client_optim_parameters_dict[f"client_optim_parameter_{idx}"])
                    selected_model_params_grad = selected_model_params_grad.float()
                    self.central_server_checker = self.central_server_checker.float()
                    selected_model_params_grad = selected_model_params_grad.to("cuda")
                    self.central_server_checker = self.central_server_checker.to("cuda")

                    dot_product = torch.dot(
                        selected_model_params_grad.view(-1),
                        self.central_server_checker.view(-1)
                    )
                    dot_product = torch.abs(dot_product).item()
                    self.dot_product.append({
                        "current_global_step": current_global_step,
                        "step": current_local_step_counter,
                        "dot_product": dot_product,
                        "current_client_id": current_client_id,
                    })
                    # print("current client: ", current_client_id)
                    # print("current_local_step_counter: ", current_local_step_counter)
                    # print("dot_product: ", dot_product)
                    # if current_local_step_counter == 100:
                    #     filename = f"{self.args.dot_product_path}_gemma_2b_{self.args.task_name}_client{current_client_id}_global_step{current_global_step}.json"
                    #     with open(filename, "w") as jsonfile:
                    #         json.dump(self.dot_product, jsonfile, indent=4)
                    #     print(f"Logs saved to {filename}")
                    server_monitor_client_optim_optimizer.step()
                    server_monitor_client_optim_optimizer.zero_grad(set_to_none=True)
                    server_monitor_client_optim_scheduler.step()

                    with torch.no_grad():
                        # with self.profiler('optimizer'):
                        for name, param in client_model_dict[f"client_model_{idx}"].named_parameters():
                            if not name in client_optim_parameters_dict[f"client_optim_parameter_{idx}"]: continue
                            param.view(-1)[self.outlier_masks[name]] = client_optim_parameters_dict[f"client_optim_parameter_{idx}"][name]
                    
                    client_optim_parameters_dict[f"client_optim_parameter_{idx}"] = {
                        n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in client_model_dict[f"client_model_{idx}"].named_parameters() if n in self.outlier_masks
                    }
                del client_optim_parameters_dict
                del client_model_dict
                torch.cuda.empty_cache()
                gc.collect() 
            return self.dot_product  
        elif self.args.training_mode == "sparse_dynamic_earlystppping":
            clients_innerproduct_list = []
            for idx, client_training_trace in enumerate(each_client_project_grad_list):
                self.dot_product = []
                current_client_id = client_training_trace["client_id"]
                client_optim_parameters_dict = {}
                client_model_dict = {}
                client_optim_parameters_dict[f"client_optim_parameter_{current_client_id}"] = copy.deepcopy(self.named_parameters_to_optim)
                client_model_dict[f"client_model_{current_client_id}"] = copy.deepcopy(self.global_model)
                server_monitor_client_optim_optimizer = torch.optim.SGD(list(client_optim_parameters_dict[f"client_optim_parameter_{current_client_id}"].values()), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
                server_monitor_client_optim_scheduler = transformers.get_constant_schedule(server_monitor_client_optim_optimizer)
                current_local_step_counter = 0
                for step_info in client_training_trace["steps"]:
                    for seed, project_grad in step_info.items():
                        torch.manual_seed(seed)
                        for n, p in client_optim_parameters_dict[f"client_optim_parameter_{current_client_id}"].items():
                            if not p.requires_grad: continue
                            if p.grad is None:
                                p.grad = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(project_grad/self.args.zo_sample)
                            else:
                                p.grad.add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=project_grad/self.args.zo_sample)
                    current_local_step_counter += 1
                    selected_model_params_grad = self.flatten_parameter_gradients(client_optim_parameters_dict[f"client_optim_parameter_{current_client_id}"])
                    selected_model_params_grad = selected_model_params_grad.float()
                    self.central_server_checker = self.central_server_checker.float()
                    selected_model_params_grad = selected_model_params_grad.to("cuda")
                    self.central_server_checker = self.central_server_checker.to("cuda")

                    dot_product = torch.dot(
                        selected_model_params_grad.view(-1),
                        self.central_server_checker.view(-1)
                    )
                    dot_product = torch.abs(dot_product).item()
                    self.dot_product.append({
                        "current_global_step": current_global_step,
                        "step": current_local_step_counter,
                        "dot_product": dot_product,
                        "current_client_id": current_client_id,
                    })
                    # print("current client: ", current_client_id)
                    # print("current_local_step_counter: ", current_local_step_counter)
                    # print("dot_product: ", dot_product)
                    # if current_local_step_counter == 100:
                    #     filename = f"{self.args.dot_product_path}_gemma_2b_{self.args.task_name}_client{current_client_id}_global_step{current_global_step}.json"
                    #     with open(filename, "w") as jsonfile:
                    #         json.dump(self.dot_product, jsonfile, indent=4)
                    #     print(f"Logs saved to {filename}")
                    server_monitor_client_optim_optimizer.step()
                    server_monitor_client_optim_optimizer.zero_grad(set_to_none=True)
                    server_monitor_client_optim_scheduler.step()

                    with torch.no_grad():
                        # with self.profiler('optimizer'):
                        for name, param in client_model_dict[f"client_model_{current_client_id}"].named_parameters():
                            if not name in client_optim_parameters_dict[f"client_optim_parameter_{current_client_id}"]: continue
                            param.view(-1)[self.outlier_masks[name]] = client_optim_parameters_dict[f"client_optim_parameter_{current_client_id}"][name]
                    
                    client_optim_parameters_dict[f"client_optim_parameter_{current_client_id}"] = {
                        n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in client_model_dict[f"client_model_{current_client_id}"].named_parameters() if n in self.outlier_masks
                    }
                clients_innerproduct_list.append(self.dot_product)
                del client_optim_parameters_dict
                del client_model_dict
                torch.cuda.empty_cache()
                gc.collect() 
            return clients_innerproduct_list
        else:
            for client_data in each_client_project_grad_list:
                print("client_data: ", client_data)
                for key, project_grad in client_data.items():
                    if key == 'client_id': continue
                    torch.manual_seed(key)
                    for name, selected_param in self.named_parameters_to_optim.items():
                        z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
                        if selected_param.grad is None:
                            selected_param.grad = z * (project_grad/self.args.zo_sample)
                        else:
                            selected_param.grad.add_(z, alpha=(project_grad/self.args.zo_sample))
            
            selected_model_params_grad = self.flatten_parameter_gradients(self.named_parameters_to_optim)
            selected_model_params_grad = selected_model_params_grad.float()
            self.central_server_checker = self.central_server_checker.float()

            selected_model_params_grad = selected_model_params_grad.to("cuda")
            self.central_server_checker = self.central_server_checker.to("cuda")

            dot_product = torch.dot(
                selected_model_params_grad.view(-1),
                self.central_server_checker.view(-1)
            )
            dot_product = torch.abs(dot_product).item()
            for name, selected_param in self.named_parameters_to_optim.items():
                selected_param.grad = None  # Clear gradients explicitly
            self.dot_product.append({
                "step": current_global_step,
                "dot_product": dot_product
            })
            print("current_global_step: ", current_global_step)
            print("dot_product: ", dot_product)
            if current_global_step == 100:
                with open(f"{self.args.dot_product_path}.json", "w") as jsonfile:
                    json.dump(self.dot_product, jsonfile, indent=4)
                print("Logs saved to dot_product.json")
            return dot_product

    
    def flatten_parameter_gradients(self, named_parameters_to_optim):
        """
        将 self.named_parameters_to_optim 中所有参数的梯度铺平并拼接成一个一维张量。

        Args:
            named_parameters_to_optim (dict): 包含参数名称和参数张量的字典。
        
        Returns:
            flat_grad (torch.Tensor): 拼接后的所有梯度的一维张量。
        """
        flat_grad = []  # 存储展平的梯度
        for name, param in named_parameters_to_optim.items():
            if param.grad is not None:  # 确保该参数有梯度
                flat_grad.append(param.grad.view(-1))  # 展平并加入列表
        if flat_grad:
            flat_grad = torch.cat(flat_grad, dim=0)  # 拼接所有展平的梯度
        else:
            flat_grad = torch.tensor([], device=list(named_parameters_to_optim.values())[0].device)  # 如果没有梯度，返回空张量
        return flat_grad
    
    # calculate loss and ppl
    @torch.no_grad()
    def model_eval(self, global_model):
        self.global_model.eval()
        losses = []
        for step, batch in enumerate(self.eval_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = self.global_model(**batch)
            loss = outputs[0]
            losses.append(loss.repeat(len(batch['input_ids'])))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            print("eval_loss: ", eval_loss)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        self.global_model.train()
        return perplexity    
    

    # calculate the accuracy
    def evaluate_accuracy(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(eval_samples):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )
        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics
    

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.global_model.device)
        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.global_model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.global_model.eval()
                logits = self.global_model(input_ids=input_ids).logits
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

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        print("=== Candidate %d (without context) SFC ===" % candidate_id)
                        print(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        print(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
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


# for per_client_proj_grad in grad_accumulator:
                    #     project_grad_accelerator = {}
                    #     # fill project_grad_accelerator
                    #     for k, v in per_client_proj_grad.items():
                    #         if k == 'client_id': continue
                    #         torch.manual_seed(k)
                    #         for n, p in self.global_model.named_parameters():
                    #             if n not in project_grad_accelerator:
                    #                 project_grad_accelerator[n] = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(v/self.args.zo_sample)
                    #             else:
                    #                 project_grad_accelerator[n].add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=v/self.args.zo_sample)
                    #     # change the grad
                    #     for n, p in self.global_model.named_parameters():
                    #         if not p.requires_grad: continue
                    #         if p.grad is None:
                    #             p.grad = project_grad_accelerator[n]/len(grad_accumulator)
                    #         else:
                    #             p.grad.add_(project_grad_accelerator[n]/len(grad_accumulator))
                    #     del project_grad_accelerator
                    # del grad_accumulator
                    # self.optimizer.step()
                    # self.optimizer.zero_grad(set_to_none=True)
                    # self.lr_scheduler.step()
                    # return self.global_model


        #                 def calculate_virtual_path(self, grad_accumulator, current_step):
        # for per_client_proj_grad in grad_accumulator:
        #     for key, value in per_client_proj_grad.items():

        #         client_id = per_client_proj_grad["client_id"]

        #         seed_keys = [k for k in per_client_proj_grad.keys() if k != "client_id"]

        #         grouped_seeds = [seed_keys[i:i+4] for i in range(0, len(seed_keys), 4)]

        #         pretrain_vec = self.central_server_checker

        #         client_results = []

        #         for group in grouped_seeds:
        #             cos_sims = []
        #             for seed in group:
        #                 seed_vec = per_client_proj_grad[seed]
        #                 seed_vec = torch.tensor(seed_vec, device=pretrain_vec.device)
        #                 # 将标量扩展为与 pretrain_vec 同形状的张量
        #                 client_estimated_grad_vector = torch.full_like(pretrain_vec, seed_vec)
        #                 print("client_estimated_grad_vector: ", client_estimated_grad_vector)
        #                 # 计算余弦相似度 = (u·v)/(||u||*||v||)
        #                 numerator = (client_estimated_grad_vector * pretrain_vec).sum()
        #                 denominator = (client_estimated_grad_vector.norm() * pretrain_vec.norm())
        #                 cos_sim = numerator / denominator
        #                 cos_sims.append(cos_sim.item())

        #             # 对每组4个seed的余弦相似度取平均
        #             if len(cos_sims) > 0:
        #                 avg_cos = sum(cos_sims) / len(cos_sims)
        #                 client_results.append(avg_cos)

        #         # 将该client的结果保存到全局results中
        #         self.path_recorder[client_id] = client_results
        #         print("path: ", self.path_recorder)
        




        # # calculate cos similarity
                    # selected_model_params_grad = self.flatten_parameter_gradients(self.named_parameters_to_optim)

                    # selected_model_params_grad = selected_model_params_grad.float()
                    # self.central_server_checker = self.central_server_checker.float()

                    # # device = selected_model_params_grad.device

                    # selected_model_params_grad = selected_model_params_grad.to("cuda")
                    # self.central_server_checker = self.central_server_checker.to("cuda")

                    # dot_product = torch.dot(
                    #     selected_model_params_grad.view(-1),
                    #     self.central_server_checker.view(-1)
                    # )
                    # dot_product = torch.abs(dot_product)
                    # norm_a = torch.norm(selected_model_params_grad)
                    # norm_b = torch.norm(self.central_server_checker)
                    # cos_sim_manual = dot_product / (norm_a * norm_b)
                    # cos_sim_manual = torch.abs(cos_sim_manual)
                    # print("Current step: ", step)
                    # print("Current dot product: ", dot_product.item())
                    # print("Cosine Similarity (manual):", cos_sim_manual.item())
                    # # print("Norm: ", norm_a.item())
                    # self.cos_similarity_log.append({"step": step, "cosine_similarity": cos_sim_manual.item()})
                    # # self.norm_log.append({"step": step, "norm": norm_a.item()})
                    # self.dot_product.append({"step": step, "dot_product": dot_product.item()})

                    # if step == 100:
                    #     # 保存日志到 JSON 文件
                    #     with open(f"{self.args.save_cossim_path}.json", "w") as jsonfile:
                    #         json.dump(self.cos_similarity_log, jsonfile, indent=4)
                    #     print("Logs saved to cossim.json")
                    #     # print("Logs saved to cosine_similarity_log.json")
                    #     with open(f"{self.args.dot_product_path}.json", "w") as jsonfile:
                    #         json.dump(self.dot_product, jsonfile, indent=4)
                    #     print("Logs saved to dot_product.json")

                    #     # with open(f"{self.args.save_norm_path}.json", "w") as jsonfile:
                    #     #     json.dump(self.norm_log, jsonfile, indent=4)
                    #     # print("Logs saved to norm_log.json")




        #             for per_client_proj_grad in each_client_project_grad_list:
        #     current_client_id = -1
        #     self.clients_score_recorder = []
        #     checker_named_parameters_to_optim = copy.deepcopy(self.named_parameters_to_optim)
        #     if self.args.use_adam:
        #         checker_optimizer = torch.optim.Adam(list(checker_named_parameters_to_optim.values()), lr=self.args.learning_rate)
        #     elif self.args.use_momentum:
        #         checker_optimizer = torch.optim.SGD(list(checker_named_parameters_to_optim.values()), lr=self.args.learning_rate, momentum=0.9)
        #     else:
        #         checker_optimizer = torch.optim.SGD(list(checker_named_parameters_to_optim.values()), lr=self.args.learning_rate)

        #     # check_lr_scheduler = transformers.get_constant_schedule(checker_optimizer)

        #     current_client_id = per_client_proj_grad['client_id']
        #     # analyze client local steps virtual path
        #     current_local_step = initial_step
        #     for client_each_step_project_grad in per_client_proj_grad['steps_data']:
        #         for seed_key, project_grad_value in client_each_step_project_grad.items():
        #             torch.manual_seed(seed_key)
        #             for name, selected_param in checker_named_parameters_to_optim.items():
        #                 z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
        #                 if selected_param.grad is None:
        #                     selected_param.grad = z * (project_grad_value/self.args.zo_sample)
        #                 else:
        #                     selected_param.grad.add_(z, alpha=(project_grad_value/self.args.zo_sample))

        #         # calculate cos similarity
        #         selected_model_params_grad = self.flatten_parameter_gradients(checker_named_parameters_to_optim)

        #         selected_model_params_grad = selected_model_params_grad.float()
        #         self.central_server_checker = self.central_server_checker.float()

        #         # device = selected_model_params_grad.device

        #         selected_model_params_grad = selected_model_params_grad.to("cuda")
        #         self.central_server_checker = self.central_server_checker.to("cuda")

        #         dot_product = torch.inner(
        #             selected_model_params_grad.view(-1),
        #             self.central_server_checker.view(-1)
        #         )
        #         dot_product = torch.abs(dot_product)

        #         norm_a = torch.norm(selected_model_params_grad)
        #         norm_b = torch.norm(self.central_server_checker)
        #         cos_sim_manual = dot_product / (norm_a * norm_b)

        #         cos_sim_manual = torch.abs(cos_sim_manual)
        #         print("current client: ", current_client_id,'current_step: ', current_local_step, 'current_global_step: ', current_global_step, 'dot_product: ', dot_product.item(), 'cos_sim: ', cos_sim_manual.item())
        #         self.clients_score_recorder.append({
        #             "client_id": current_client_id,
        #             "step": current_local_step,
        #             "global_step": current_global_step,
        #             "dot_product": dot_product.item(),
        #             "cos_sim": cos_sim_manual.item(),
        #         })
        #         current_local_step += 1

        #         checker_optimizer.step()
        #         checker_optimizer.zero_grad(set_to_none=True)
        #         # check_lr_scheduler.step()

        #     del checker_named_parameters_to_optim
        #     del checker_optimizer
        #     # del check_lr_scheduler

        #     gc.collect()
        #     torch.cuda.empty_cache()

        #     # Save to JSON file
        #     file_name = f"{self.args.task_name}_client_{current_client_id}_global_step_{current_global_step}.json"
        #     save_path = os.path.join("/home/yran1/FlZO_Vritual_Path", file_name)  # Ensure output_dir is set in args
        #     with open(save_path, "w") as json_file:
        #         json.dump(self.clients_score_recorder, json_file, indent=4)

        # return self.clients_score_recorder









                #             for name in client_optim_parameters_dict[f"client_optim_parameter_{idx}"].keys():
                #         client_optim_parameters_dict[f"client_optim_parameter_{idx}"][name] = client_optim_parameters_dict[f"client_optim_parameter_{idx}"][name].to("cpu")
                #     del server_monitor_client_optim_optimizer
                #     del server_monitor_client_optim_scheduler
                #     gc.collect()
                # for name in self.named_parameters_to_optim.keys():
                #     self.named_parameters_to_optim[name] = self.named_parameters_to_optim[name].to("cpu")
                
                # for name in self.named_parameters_to_optim.keys():
                #     self.named_parameters_to_optim[name].data.zero_()
                
                # for i in range(self.args.total_num_users):
                #     client_key = f"client_optim_parameter_{i}"
                #     for name in self.named_parameters_to_optim.keys():
                #         self.named_parameters_to_optim[name].data.add_(client_optim_parameters_dict[client_key][name])
                
                # for name in self.named_parameters_to_optim.keys():
                #     self.named_parameters_to_optim[name].data.div_(self.args.total_num_users)
                
                # for name in self.named_parameters_to_optim.keys():
                #     self.named_parameters_to_optim[name] = self.named_parameters_to_optim[name].to(device)
                
                # with torch.no_grad():
                #     # with self.profiler('optimizer'):
                #     for name, param in self.global_model.named_parameters():
                #         if not name in self.named_parameters_to_optim: continue
                #         param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]

                # del client_optim_parameters_dict
                # del grad_accumulator
                # gc.collect()
                # return self.global_model, self.named_parameters_to_optim