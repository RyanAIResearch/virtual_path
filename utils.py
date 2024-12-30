import json
import os
import contextlib
from typing import Optional, Union
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
import logging
import time
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
import transformers
from typing import Optional, Union, List, Dict, Any
import signal
from subprocess import call
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
InputDataClass = NewType("InputDataClass", Any)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import datasets
import inspect
import random



logger = logging.getLogger(__name__)


def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=None, **kwargs):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
      last option_len tokens 
    - num_options: a list of int indicating the number of options for each example (this will be #label
      words for classification tasks and #choices for multiple choice tasks), and a classification loss
      will be calculated.
    """
    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    if labels is None:
        return outputs
    logits = outputs.logits

    loss = None
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # Apply option len (do not calculate loss on the non-option part)
    for _i, _len in enumerate(option_len):
        shift_labels[_i, :-_len] = -100

    # Calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    if num_options is not None: 
        # Train as a classification tasks
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100 # Option part
        shift_labels[~mask] = 0 # So that it doesn't mess up with indexing

        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) # (bsz x num_options, len)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1) # (bsz x num_options)

        if any([x != num_options[0] for x in num_options]):
            # Multi choice tasks with different number of options
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0) # (1, num_options)
                _labels = labels[start_id:end_id][0].unsqueeze(0) # (1)
                loss = loss_fct(_logits, _labels) + loss
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            num_options = num_options[0]
            selected_log_probs = selected_log_probs.view(-1, num_options) # (bsz, num_options)
            labels = labels.view(-1, num_options)[:, 0] # Labels repeat so we only take the first one
            loss = loss_fct(selected_log_probs, labels)
    else:
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def forward_wrap_with_option_len_SofiaG(model, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=None, attention_mask=None, logits=None, model_optimization=False):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
    last option_len tokens 
    - num_options: a list of int indicating the number of options for each example (this will be #label
    words for classification tasks and #choices for multiple choice tasks), and a classification loss
    will be calculated.
    """
    # 
    selected_log_probs = logits
    if selected_log_probs is None:
        label_word_tokens = [6587, 372]
        B, L = input_ids.shape
        input_ids = input_ids.reshape(B // 2, 2, L)[:, 0, :-1]
        labels = labels.reshape(B // 2, 2)[:, 0]
        attention_mask = attention_mask.reshape(B // 2, 2, L)[:, 0, :-1]
        option_len = option_len[:B//2]
        num_options = num_options[:B//2]
        outputs = model.original_forward(input_ids=input_ids, attention_mask=attention_mask)
        if labels is None:
            return outputs
        logits = outputs.logits

        loss = None
        # Shift so that tokens < n predict n
        shift_logits = logits.contiguous()
        # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
        shift_labels = torch.clone(input_ids).contiguous()
        shift_labels[shift_labels == model.config.pad_token_id] = -100

        # Apply option len (do not calculate loss on the non-option part)
        for _i, _len in enumerate(option_len):
            shift_labels[_i, :-_len] = -100

        # Calculate the loss
        # Train as a classification tasks
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100 # Option part
        shift_labels[~mask] = 0 # So that it doesn't mess up with indexing

        selected_log_probs = log_probs[:, -1, label_word_tokens]
    
    if model_optimization:
        loss = F.cross_entropy(selected_log_probs, labels)
    else:
        samp_dist = torch.distributions.Categorical(logits=selected_log_probs)
        y_hat_sample = samp_dist.sample()
        loss = F.cross_entropy(selected_log_probs, y_hat_sample)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=selected_log_probs,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def encode_prompt(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False, generation=False, generation_with_gold=False, max_new_tokens=None):
    """
    Encode prompts for eval_sample
    Input: 
    - task, template: task and template class
    - train_samples, eval_sample: demonstrations and the actual sample
    - tokenizer, max_length: tokenizer and max length
    - sfc: generate prompts for calibration (surface form competition; https://arxiv.org/abs/2104.08315)
    - icl_sfc: generate prompts for ICL version calibration
    - generation: whether it is an generation task
    - generation_with_gold: whether to include the generation-task gold answers (for training)
    - max_new_tokens: max number of new tokens to generate so that we can save enough space 
      (only for generation tasks)
    Output:
    - encodings: a list of N lists of tokens. N is the number of options for classification/multiple-choice.
    - option_lens: a list of N integers indicating the number of option tokens.
    """
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
        logger.warn("Exceed max length")
    if tokenizer.bos_token or tokenizer.add_bos_token:
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]  
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]
    return encodings, option_lens


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
 


@dataclass
class ICLCollator:
    """
    Collator for ICL
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        
        pad_id = self.tokenizer.pad_token_id

        pad_ids = {"input_ids": pad_id, "attention_mask": 0, "sfc_input_ids": pad_id, "sfc_attention_mask": 0, "labels": pad_id}
        for key in first:
            pp = pad_ids[key]
            lens = [len(f[key]) for f in features]
            max_len = max(lens)
            feature = np.stack([np.pad(f[key], (0, max_len - lens[i]), "constant", constant_values=(0, pp)) for i, f in enumerate(features)])
            padded_feature = torch.from_numpy(feature).long()
            batch[key] = padded_feature
            
        return batch


@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator for training
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [ff for f in features for ff in f]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


@dataclass
class NondiffCollator(DataCollatorMixin):
    """
    Collator for non-differentiable objectives
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name and k != "gold"} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        if "gold" in features[0]:
            batch["gold"] = [feature["gold"] for feature in features]
        
        return batch
        

class SIGUSR1Callback(transformers.TrainerCallback):
    """
    This callback is used to save the model when a SIGUSR1 signal is received
    (SLURM stop signal or a keyboard interruption signal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Signal received")

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]


@contextlib.contextmanager
def count_time(name):
    print("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        print("Done with %.2fs" % (time.time() - start_time))


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


def write_predictions_to_file(final_preds, output):
    with open(output, "w") as f:
        for pred in final_preds:
            f.write(json.dumps(pred, cls=EnhancedJSONEncoder) + "\n")


def write_metrics_to_file(metrics, output):
    json.dump(metrics, open(output, "w"), cls=EnhancedJSONEncoder, indent=4)


def dataset_preprocessing(args, task, tokenizer, train_samples, eval_samples, test_samples):
    tokenizer.padding_side = "left"

    class HFDataset(Dataset):

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]
        
        def __get__(self):
            return self.data


    def _convert(samples, task, args, tokenizer):
        """
        Convert samples to HF-compatible dataset
        """
        data = []
        for sample in samples:
            encoded_candidates, option_lens = encode_prompt(
                task, task.get_template(), [], sample, tokenizer, 
                max_length=args.max_length, generation=task.generation, generation_with_gold=True, 
                max_new_tokens=args.max_new_tokens
            )
            if task.generation:
                correct_candidate_id = 0
            elif isinstance(sample.correct_candidate, list):
                correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
            else:
                correct_candidate_id = sample.candidates.index(sample.correct_candidate)
            
            if args.non_diff:
                # For non-differentiable objective, there is no teacher forcing thus the 
                # current answer part is removed
                encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

            if args.train_as_classification:
                # For classification, we provide the label as the correct candidate id
                data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
            elif args.only_train_option:
                print("is use only_train_option?")
                # Otherwise, it is just LM-style teacher forcing
                if args.non_diff:
                    # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
            else:
                data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
        return data

    with count_time("Tokenizing training samples"):
        train_dataset = HFDataset(_convert(train_samples, task, args, tokenizer))
        eval_dataset = HFDataset(_convert(eval_samples, task, args, tokenizer))  
        if test_samples:
            test_dataset = HFDataset(_convert(test_samples, task, args, tokenizer)) 
            eval_data_for_trainer = {"dev": eval_dataset, "test": test_dataset}
            args.metric_for_best_model = 'eval_dev_loss'
        else:
            eval_data_for_trainer = eval_dataset
    return train_dataset, eval_dataset, test_dataset


def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    # if is_torch_available():
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


# Data Processing       
def get_train_dataloader(args, train_dataset, data_collator, local_model, _signature_columns, accelerator) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = train_dataset
    data_collator = data_collator
    default_label_names = find_labels(local_model.__class__)

    if isinstance(train_dataset, datasets.Dataset):
        train_dataset = _remove_unused_columns(local_model, _signature_columns, train_dataset, default_label_names, description="training")
    else:
        data_collator = _get_collator_with_removed_columns(args, data_collator, _signature_columns, local_model, default_label_names, description="training")
                                                        
    dataloader_params = {
        "batch_size": args.per_device_train_batch_size,
        "collate_fn": data_collator,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": 0,
    }
    print("isinstance(train_dataset, torch.utils.data.IterableDataset): ", isinstance(train_dataset, torch.utils.data.IterableDataset))
    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = _get_train_sampler(train_dataset)
        dataloader_params["drop_last"] = False
        dataloader_params["worker_init_fn"] = seed_worker

    return accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

def _get_collator_with_removed_columns(
        args, data_collator: Callable, _signature_columns, model, label_names, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not args.remove_unused_columns:
            return data_collator
        _set_signature_columns_if_needed(_signature_columns, label_names, model)
        signature_columns = _signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=model.__class__.__name__,
        )
        return remove_columns_collator

def _set_signature_columns_if_needed(_signature_columns, label_names, model):
    if _signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        model_to_inspect = model
        signature = inspect.signature(model_to_inspect.forward)
        _signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        _signature_columns += list(set(["label", "label_ids"] + label_names))


def _get_train_sampler(train_dataset) -> Optional[torch.utils.data.Sampler]:
    if train_dataset is None:
        return None

    return RandomSampler(train_dataset)

def _get_eval_sampler(eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
    return SequentialSampler(eval_dataset)


def _remove_unused_columns(local_model, _signature_columns, dataset: "datasets.Dataset", label_names, description: Optional[str] = None):
    _set_signature_columns_if_needed(_signature_columns, label_names, local_model)
    signature_columns = _signature_columns

    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    if len(ignored_columns) > 0:
        dset_description = "" if description is None else f"in the {description} set"
        print(
            f"The following columns {dset_description} don't have a corresponding argument in "
            f"`{local_model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            f" If {', '.join(ignored_columns)} are not expected by `{local_model.__class__.__name__}.forward`, "
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

def _set_signature_columns_if_needed(_signature_columns, label_names, local_model):
    if _signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        model_to_inspect = local_model
        signature = inspect.signature(model_to_inspect.forward)
        _signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        _signature_columns += list(set(["label", "label_ids"] + label_names))

def infer_framework(model_class):
    """
    Infers the framework of a given model without using isinstance(), because we cannot guarantee that the relevant
    classes are imported or available.
    """
    for base_class in inspect.getmro(model_class):
        module = base_class.__module__
        name = base_class.__name__
        if module.startswith("tensorflow") or module.startswith("keras") or name == "TFPreTrainedModel":
            return "tf"
        elif module.startswith("torch") or name == "PreTrainedModel":
            return "pt"
        elif module.startswith("flax") or module.startswith("jax") or name == "FlaxPreTrainedModel":
            return "flax"
    else:
        raise TypeError(f"Could not infer framework from class {model_class}.")

def find_labels(model_class):
    """
    Find the labels used by a given model.

    Args:
        model_class (`type`): The class of the model.
    """
    model_name = model_class.__name__
    framework = infer_framework(model_class)
    if framework == "tf":
        signature = inspect.signature(model_class.call)  # TensorFlow models
    elif framework == "pt":
        signature = inspect.signature(model_class.forward)  # PyTorch models
    else:
        signature = inspect.signature(model_class.__call__)  # Flax models

    if "QuestionAnswering" in model_name:
        return [p for p in signature.parameters if "label" in p or p in ("start_positions", "end_positions")]
    else:
        return [p for p in signature.parameters if "label" in p]


def get_test_dataloader(args, test_dataset: Dataset, data_collator, local_model, _signature_columns, accelerator) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = data_collator

        default_label_names = find_labels(local_model.__class__)

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset = _remove_unused_columns(local_model, _signature_columns, test_dataset, default_label_names, description="test")
        else:
            data_collator = _get_collator_with_removed_columns(args, data_collator, _signature_columns, local_model, default_label_names, description="test")

        dataloader_params = {
            "batch_size": args.per_device_eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": 0,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = _get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = args.dataloader_prefetch_factor

        # We use the same batch_size as for eval.
        return accelerator.prepare(DataLoader(test_dataset, **dataloader_params))