import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import math

def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class LoRALinear(nn.Linear):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False, # Not sure if this will affect saving/loading models so just set it to be False
        dtype = None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        self.dtype = dtype
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features), dtype=torch.float32))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r), dtype=torch.float32))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += ((self.lora_dropout(x.to(dtype=self.lora_A.dtype)) @ \
                           self.lora_A.transpose(0, 1) @ \
                            self.lora_B.transpose(0, 1)) * self.scaling).to(dtype=result.dtype)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


# class LoRA:

#     def __init__(self, model, r, alpha, need_all_linear=False):
#         """
#         Input:
#         r, alpha: LoRA hyperparameters
#         """

#         self.model = model
#         self.hidden_dim = model.config.hidden_size

#         if model.config.model_type == "opt":
#             attention_name = "attn"
#             mlp_name = ''
#         elif model.config.model_type == 'llama':
#             attention_name = "attn"            
#             mlp_name = 'mlp'
#         elif model.config.model_type == "roberta":
#             attention_name = "attention"
#         else:
#             raise NotImplementedError
#         # Insert LoRA
#         for key, _ in model.named_modules():
#             if key[-len(attention_name):] == attention_name:
#                 print(f"Inject lora to: {key}")
#                 _, _, attn = find_module(model, key)
#                 if model.config.model_type in ["opt", 'llama']:
#                     original_q_weight = attn.q_proj.weight.data
#                     if attn.q_proj.bias is not None:
#                         original_q_bias = attn.q_proj.bias.data
#                     original_v_weight= attn.v_proj.weight.data
#                     if attn.v_proj.bias is not None:
#                         original_v_bias = attn.v_proj.bias.data
#                     attn.q_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=(attn.q_proj.bias is not None), dtype=original_q_weight.dtype).to(original_q_weight.device)
#                     attn.v_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=(attn.v_proj.bias is not None), dtype=original_v_weight.dtype).to(original_v_weight.device)
#                     attn.q_proj.weight.data = original_q_weight 
#                     if attn.q_proj.bias is not None:
#                         attn.q_proj.bias.data = original_q_bias
#                     attn.v_proj.weight.data = original_v_weight
#                     if attn.v_proj.bias is not None:
#                         attn.v_proj.bias.data = original_v_bias
                    
#                     if need_all_linear:
#                         original_k_weight = attn.k_proj.weight.data
#                         original_o_weight = attn.o_proj.weight.data
#                         if attn.k_proj.bias is not None:
#                             original_k_bias = attn.k_proj.bias.data
#                         if attn.o_proj.bias is not None:
#                             original_o_bias = attn.o_proj.bias.data
                        
#                         attn.k_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=(attn.k_proj.bias is not None), dtype=original_k_weight.dtype).to(original_k_weight.device)
#                         attn.o_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha, bias=(attn.o_proj.bias is not None), dtype=original_o_weight.dtype).to(original_o_weight.device)
                        
#                         attn.k_proj.weight.data = original_k_weight 
#                         if attn.k_proj.bias is not None:
#                             attn.k_proj.bias.data = original_k_bias
#                         attn.o_proj.weight.data = original_o_weight
#                         if attn.o_proj.bias is not None:
#                             attn.o_proj.bias.data = original_o_bias
                    
#                 else:
#                     raise NotImplementedError
        
#             if need_all_linear and key[-len(mlp_name):] == mlp_name:
#                 print(f"Inject lora to: {key}")
#                 _, _, FFN = find_module(model, key)
#                 if model.config.model_type == 'llama':
#                     original_down_proj_weight = attn.down_proj.weight.data
#                     if attn.down_proj.bias is not None:
#                         original_down_bias = attn.q_proj.bias.data
#                     original_v_weight= attn.v_proj.weight.data
#                     if attn.v_proj.bias is not None:
#                         original_v_bias = attn.v_proj.bias.data

#         # Freeze non-LoRA parameters
#         for n, p in model.named_parameters():
#             if "lora" not in n:
#                 p.requires_grad = False

class LoRA:

    def __init__(self, model, r, alpha, need_all_linear=True):
        """
        Input:
        r, alpha: LoRA hyperparameters
        """
        self.model = model
        self.hidden_dim = model.config.hidden_size

        if model.config.model_type == "opt":
            transformer_layers = model.model.decoder.layers
            if need_all_linear:
                linear_names = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj', 'self_attn.out_proj', 'fc1', 'fc2']
            else:
                linear_names = ['self_attn.q_proj', 'self_attn.v_proj']

        elif model.config.model_type in ['llama', 'mistral']:
            transformer_layers = model.model.layers
            if need_all_linear:
                linear_names = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
            else:
                linear_names = ['self_attn.q_proj', 'self_attn.v_proj']

        else:
            raise NotImplementedError()
        
        # Insert LoRA
        for i, layer in enumerate(transformer_layers):
            layer_modules = {n : module for n, module in layer.named_modules()}
            print(f"Inject lora to layer {i}")
            
            for linear_name in linear_names:
                linear_module = layer_modules[linear_name]
                original_weight = linear_module.weight.data
                if linear_module.bias is not None:
                    original_bias = linear_module.bias.data

                in_features = linear_module.weight.shape[1]
                out_features = linear_module.weight.shape[0]
                lora_linear_module = LoRALinear(in_features, out_features, r=r, lora_alpha=alpha, bias=(linear_module.bias is not None), dtype=original_weight.dtype).to(original_weight.device)
                if lora_linear_module.bias is not None:
                    lora_linear_module.bias.data = original_bias
                layer_modules[linear_name] = lora_linear_module
            
            if model.config.model_type == "opt":
                q_weight_data = layer.self_attn.q_proj.weight
                k_weight_data = layer.self_attn.k_proj.weight
                v_weight_data = layer.self_attn.v_proj.weight
                o_weight_data = layer.self_attn.out_proj.weight
                fc1_weight_data = layer.fc1.weight
                fc2_weight_data = layer.fc2.weight

                if need_all_linear:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.q_proj.weight.data = q_weight_data

                    layer.self_attn.k_proj = layer_modules['self_attn.k_proj']
                    layer.self_attn.k_proj.weight.data = k_weight_data

                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']
                    layer.self_attn.v_proj.weight.data = v_weight_data

                    layer.self_attn.out_proj = layer_modules['self_attn.out_proj']
                    layer.self_attn.out_proj.weight.data = o_weight_data

                    layer.fc1 = layer_modules['fc1']
                    layer.fc1.weight.data = fc1_weight_data

                    layer.fc2 = layer_modules['fc2']
                    layer.fc2.weight.data = fc2_weight_data

                else:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.q_proj.weight.data = q_weight_data

                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']
                    layer.self_attn.v_proj.weight.data = v_weight_data

            elif model.config.model_type in ['llama', 'mistral']:
                if need_all_linear:
                    q_weight_data = layer.self_attn.q_proj.weight
                    k_weight_data = layer.self_attn.k_proj.weight
                    v_weight_data = layer.self_attn.v_proj.weight
                    o_weight_data = layer.self_attn.o_proj.weight
                    gate_weight_data = layer.mlp.gate_proj.weight
                    up_weight_data = layer.mlp.up_proj.weight
                    down_weight_data = layer.mlp.down_proj.weight

                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.q_proj.weight.data = q_weight_data

                    layer.self_attn.k_proj = layer_modules['self_attn.k_proj']
                    layer.self_attn.k_proj.weight.data = k_weight_data

                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']
                    layer.self_attn.v_proj.weight.data = v_weight_data

                    layer.self_attn.o_proj = layer_modules['self_attn.o_proj']
                    layer.self_attn.o_proj.weight.data = o_weight_data

                    layer.mlp.gate_proj = layer_modules['mlp.gate_proj']
                    layer.mlp.gate_proj.weight.data = gate_weight_data

                    layer.mlp.up_proj = layer_modules['mlp.up_proj']
                    layer.mlp.up_proj.weight.data = up_weight_data

                    layer.mlp.down_proj = layer_modules['mlp.down_proj']
                    layer.mlp.down_proj.weight.data = down_weight_data

                else:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.q_proj.weight.data = q_weight_data

                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']
                    layer.self_attn.v_proj.weight.data = v_weight_data

            else:
                raise NotImplementedError()


        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
