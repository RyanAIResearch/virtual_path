import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import torch
import numpy as np

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



