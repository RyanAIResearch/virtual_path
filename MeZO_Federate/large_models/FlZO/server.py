# -*- coding: UTF-8 -*-
#
# ProjectName: PromptZO-master
# Description: 
# Author: 

import pandas as pd
import numpy as np

import torch, math
from torch import nn
from transformers import default_data_collator
import transformers
import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping

class Server(object):

    def __init__(self, args, train_dataloader, test_dataloader, global_model, named_parameters_to_optim, outlier_masks, optimizer, lr_scheduler):

        self.args = args
        
        self.global_model = global_model

        self.test_dataloader = test_dataloader

        self.train_dataloader = train_dataloader

        self._train_batch_size = self.train_dataloader.batch_size

        self.named_parameters_to_optim = named_parameters_to_optim

        self.outlier_masks = outlier_masks

        self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler

        self.seed_lists = [np.random.randint(10000000) for _ in range(4)]

        print("seed_lists: ", self.seed_lists)

    
    def get_seed_list(self):
        return self.seed_lists
    

    def model_aggregate(self, grad_accumulator):
        with torch.no_grad():
            self.global_model.eval()
            if self.args.training_mode == "loss_single_batch":
                for per_client_proj_grad in grad_accumulator:
                    project_grad_accelerator = {}
                    # fill project_grad_accelerator
                    for k, v in per_client_proj_grad.items():
                        if k == 'clinet_id': continue
                        torch.manual_seed(k)
                        for n, p in self.global_model.named_parameters():
                            if n not in project_grad_accelerator:
                                project_grad_accelerator[n] = torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype).mul_(v/self.args.zo_sample)
                            else:
                                project_grad_accelerator[n].add_(torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype), alpha=v/self.args.zo_sample)
                    # change the grad
                    for n, p in self.global_model.named_parameters():
                        if not p.requires_grad: continue
                        if p.grad is None:
                            p.grad = project_grad_accelerator[n]/len(grad_accumulator)
                        else:
                            p.grad.add_(project_grad_accelerator[n]/len(grad_accumulator))
                    del project_grad_accelerator
                del grad_accumulator
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                return self.global_model
            
            elif self.args.training_mode == "loss_full_batch":
                for per_client_proj_grad in grad_accumulator:
                    project_grad_accelerator = {}
                    # fill the grad
                    for k, v in per_client_proj_grad.items():
                        if k == 'clinet_id': continue
                        torch.manual_seed(k)
                        for name, selected_param in self.named_parameters_to_optim.items():
                            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
                            selected_param.grad = v * z

                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        with torch.no_grad():
                            # with self.profiler('optimizer'):
                            for name, param in self.global_model.named_parameters():
                                if not name in self.named_parameters_to_optim: continue
                                param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]
                        
                        del self.optimizer
                        del self.outlier_masks
                        del self.named_parameters_to_optim

                        self.outlier_masks = self.get_gradient_masks(self.global_model, self.args.outlier_percentage)

                        self.named_parameters_to_optim = {
                            n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in self.global_model.named_parameters() if n in self.outlier_masks
                        }
                        self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)

                return self.global_model, self.named_parameters_to_optim, self.outlier_masks
                        
















                        # for name, selected_param in self.named_parameters_to_optim.items():
                        #     if name not in grad_recorder:
                        #         grad_recorder[name] = selected_param
                        #         z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
                        #         selected_param.grad = z.mul_(v/4)
                        #     else:
                        #         z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
                        #         selected_param.grad.add_((z), alpha=v/4)

                # Average
                # for name, selected_param in self.named_parameters_to_optim.items():
                #     selected_param.grad = selected_param.grad / len(grad_accumulator)

                # self.optimizer.step()
                # self.optimizer.zero_grad(set_to_none=True)
                # torch.cuda.empty_cache()

                # with torch.no_grad():
                #     # with self.profiler('optimizer'):
                #     for name, param in self.global_model.named_parameters():
                #         if not name in self.named_parameters_to_optim: continue
                #         param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name] / len(grad_accumulator)
                # print("percentage: ", self.args.outlier_percentage)
                # del self.optimizer
                # del grad_recorder
                # del self.outlier_masks
                # del self.named_parameters_to_optim

                # self.outlier_masks = self.get_gradient_masks(self.global_model, self.args.outlier_percentage)

                # self.named_parameters_to_optim = {
                #     n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in self.global_model.named_parameters() if n in self.outlier_masks
                # }
                # self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)

                # return self.global_model, self.named_parameters_to_optim, self.outlier_masks


    # Get All Types Of Mask
        
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
            for sampled_batch in self.train_dataloader:
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
            for sampled_batch in self.train_dataloader:
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
            for sampled_batch in self.train_dataloader:  
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
    def model_eval(self, global_model):
        self.global_model.eval()
        losses = []
        for step, batch in enumerate(self.test_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = self.compute_loss(self.global_model, batch)
            losses.append(loss.repeat(len(batch['input_ids'])))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        self.global_model.train()
        return perplexity
    
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
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
