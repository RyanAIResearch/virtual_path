# -*- coding: UTF-8 -*-
#
# ProjectName: PromptZO-master
# Description: 
# Author: 

import pandas as pd
import numpy as np
import inspect
import torch
import transformers
from transformers import default_data_collator, DataCollatorWithPadding, PreTrainedTokenizerBase, SequenceFeatureExtractor
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import datasets
import math
import contextlib
from collections.abc import Mapping
from accelerate import Accelerator
from .training_utils import (
    find_labels,
    seed_worker
)

class Client(object):

    def __init__(
            self, 
            args, 
            local_train_data, 
            tokenizer, 
            data_collator, 
            has_mask, 
            resume_from_checkpoint, 
            named_parameters_to_optim, 
            outlier_masks, 
            epochs_trained, 
            client_model=None, 
            seed_list=None, 
            id=-1
    ):

        self.args = args

        self.local_model = client_model

        self.client_id = id

        self.train_dataset = local_train_data

        self.received_seed_list = seed_list

        self.tokenizer = tokenizer

        self.has_mask = has_mask

        self.resume_from_checkpoint = resume_from_checkpoint

        self.named_parameters_to_optim = named_parameters_to_optim

        self.epochs_trained = epochs_trained

        self.outlier_masks = outlier_masks

        self.accelerator = Accelerator()

        default_collator = data_collator

        self.data_collator = default_collator

        self._signature_columns = None

        self.execute_step = 0

        default_label_names = find_labels(self.local_model.__class__)

        self.label_names = default_label_names

        self.train_dataloader = self.get_train_dataloader()

        self.total_step = len(self.train_dataloader)

        self.is_deepspeed_enabled = False

        self.use_cpu_amp = False

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
        print("batch_size: ", self.args.per_device_train_batch_size)
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
    

    def receive_server_info(self, receive_seed_list, client_model):
        self.received_seed_list = receive_seed_list
        self.local_model = client_model

    
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
    
    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    @torch.inference_mode()
    def set_grad_with_mask(self, model, inputs, mask_dict, random_seed):

        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)

        loss1 = self.zo_forward(model, inputs)

        self.zo_perturb_parameters_with_mask(scaling_factor=-2, random_seed=random_seed, mask_dict=mask_dict)

        loss2 = self.zo_forward(model, inputs)

        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)

        global_projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        return global_projected_grad
        # torch.manual_seed(random_seed)
        # for name, selected_param in self.named_parameters_to_optim.items():
        #     z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
        #     selected_param.grad = global_projected_grad * z
        # return loss1
        

    @torch.inference_mode()
    def zo_perturb_parameters_with_mask(self, random_seed=None, mask_dict=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.local_model.named_parameters():
            if not name in self.named_parameters_to_optim: continue
            selected_param = self.named_parameters_to_optim[name]
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            param.view(-1)[mask_dict[name]] += (scaling_factor * self.args.zo_eps) * z
            del z

    
    # client training
    def local_train(self):
        completed_steps = 0
        logging_interval = 10
        eval_steps = 50
        cur_loss = 0
        losses = []
        ppls = []
        
        # Train!
        print("***** Running training *****")
        if self.args.training_mode == "loss_single_batch":
            progress_bar = tqdm(range(len(self.received_seed_list)))
        elif self.args.training_mode == "weight_full_batch":
            progress_bar = tqdm(range(self.total_step))
        elif self.args.training_mode == "loss_full_batch":
            progress_bar = tqdm(range(self.total_step))
        cur_loss = 0
        project_grad_collector = {"clinet_id" : self.client_id}

        if self.args.training_mode == "loss_single_batch":

            with torch.no_grad():
                for step, inputs in enumerate(self.train_dataloader):
                    if step < self.execute_step: continue
                    self.local_model.eval()
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    zo_sample = self.args.zo_sample
                    for _ in range(zo_sample):
                        seed = np.random.randint(10000000)

                        project_grad = self.set_grad_with_mask(self.local_model, inputs, self.outlier_masks ,seed)

                        project_grad_collector[seed] = project_grad

                        self.execute_step += 1

                        progress_bar.update(1)

                        if self.execute_step > self.total_step-1:
                            self.execute_step = 0
                    break
                return project_grad_collector
        
        elif self.args.training_mode == "loss_full_batch":
            with torch.no_grad():
                for step, inputs in enumerate(self.train_dataloader):
                    self.local_model.eval()
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    # for seed in self.received_seed_list:
                    seed = np.random.randint(10000000)
                    project_grad = self.set_grad_with_mask(self.local_model, inputs, self.outlier_masks ,seed)
                    project_grad_collector[seed] = project_grad
                    progress_bar.update(1)
                return project_grad_collector

                    