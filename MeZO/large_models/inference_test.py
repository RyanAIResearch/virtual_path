import logging
import transformers
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
        weight=None,
        bias=None, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        device = None,
        dtype = None,
        profiler=None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, device=device, dtype=dtype, **kwargs)
        if weight is not None:
            self.weight.data.copy_(weight.data)
        if bias is not None:
            self.bias.data.copy_(bias.data)

        self.profiler = profiler
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_A = nn.Parameter(self.weight.new_zeros((in_features, r), dtype=dtype, device=device))
        self.lora_B = nn.Parameter(self.weight.new_zeros((r, out_features), dtype=dtype, device=device))
        self.scaling = self.lora_alpha / self.r
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)
        with self.profiler('lora'):
            result.add_(
                torch.linalg.multi_dot(
                    [
                        x.squeeze_(), 
                        self.lora_A, 
                        self.lora_B
                    ]
                ), alpha=self.scaling)
        return result



class LoRA:

    def __init__(self, model, r, alpha, profiler, need_all_linear=True):
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

                in_features = linear_module.weight.shape[1]
                out_features = linear_module.weight.shape[0]
                lora_linear_module = LoRALinear(in_features, out_features, 
                                                weight=linear_module.weight, 
                                                bias=linear_module.bias, 
                                                r=r, lora_alpha=alpha, 
                                                device=original_weight.device,
                                                dtype=original_weight.dtype, profiler=profiler)

                layer_modules[linear_name] = lora_linear_module
            
            if model.config.model_type == "opt":
                if need_all_linear:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.k_proj = layer_modules['self_attn.k_proj']
                    layer.self_attn.v_proj = layer_modules['self_attn.k_proj']
                    layer.self_attn.out_proj = layer_modules['self_attn.out_proj']
                    layer.fc1 = layer_modules['fc1']
                    layer.fc2 = layer_modules['fc2']

                else:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']

            elif model.config.model_type in ['llama', 'mistral']:
                if need_all_linear:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.k_proj = layer_modules['self_attn.k_proj']
                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']
                    layer.self_attn.o_proj = layer_modules['self_attn.o_proj']
                    layer.mlp.gate_proj = layer_modules['mlp.gate_proj']
                    layer.mlp.up_proj = layer_modules['mlp.up_proj']
                    layer.mlp.down_proj = layer_modules['mlp.down_proj']

                else:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']

            else:
                raise NotImplementedError()


        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False



class D_Plus_S_Linear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        percentage: float=1e-3,
        weight=None,
        bias=None, 
        C4_grad_sq_block=None,
        device=None,
        dtype=None,
        profiler=None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, device=device, dtype=dtype, **kwargs)
        self.weight.data.copy_(weight.data)
        
        if bias is not None:
            self.bias.data.copy_(bias.data)
        numel = self.weight.numel()
        sparse_num = int(numel * percentage)
        if C4_grad_sq_block is None:
            # random sparsity
            nnz_indices = torch.randperm(numel, device=self.weight.device)[:sparse_num].clone()            
        else:
            nnz_indices = torch.topk(C4_grad_sq_block.view(-1), sparse_num).indices.to(self.weight.device)
            
        self.sparse_weight = torch.zeros_like(self.weight.data)
        self.sparse_weight.view(-1)[nnz_indices] = self.weight.view(-1)[nnz_indices]
        self.weight.view(-1)[nnz_indices] = 0
        self.sparse_weight_transpose = self.sparse_weight.clone().T.to_sparse_csr()
        self.sparse_weight = self.sparse_weight.to_sparse_csr()
        
        self.profiler = profiler
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.sparse_weight = torch.nn.Parameter(self.sparse_weight)
        

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        w = self.weight
        with self.profiler('sparse-add'):
            w = w + self.sparse_weight
        result = F.linear(x, w, bias=self.bias)
        if (not hasattr(self.config, 'disable_sparse_mm')) or (not self.config.disable_sparse_mm):            
            r1, r2, r3 = result.shape[0], result.shape[1], result.shape[2]
            x1, x2, x3 = x.shape[0], x.shape[1], x.shape[2]
            if x1 == 1:
                with self.profiler('sparse-mm'):
                    torch.sparse.addmm(
                        result.view(r2, r3), 
                        x.view(x2, x3), 
                        self.sparse_weight_transpose
                    ) # just for recording time
            else:
                with self.profiler('sparse-mm'):
                    torch.sparse.addmm(
                        result.view(r1 * r2, r3), 
                        x.view(x1 * x2, x3), 
                        self.sparse_weight_transpose
                    ) # just for recording time
        return result
        # return F.linear(x, (self.weight + self.sparse_weight), bias=self.bias)


class SparseTuning:

    def __init__(self, model, percentage, C4_grad_sq_list, profiler, need_all_linear=True, use_sensitive=True):
        self.model = model
        self.hidden_dim = model.config.hidden_size

        if model.config.model_type == "opt":
            transformer_layers = model.model.decoder.layers
            C4_grad_name = ["q", "k", "v", "o", "up", "down"]
            if need_all_linear:
                linear_names = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj', 'self_attn.out_proj', 'fc1', 'fc2']
            else:
                linear_names = ['self_attn.q_proj', 'self_attn.v_proj']
            if use_sensitive:
                C4_grad_name_lookup_dict = {
                    linear_names[i]: C4_grad_name[i] for i in range(7)
                }

        elif model.config.model_type in ['llama', 'mistral']:
            transformer_layers = model.model.layers
            C4_grad_name = ["q", "k", "v", "o", "gate", "up", "down"]
            if need_all_linear:
                linear_names = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
            else:
                linear_names = ['self_attn.q_proj', 'self_attn.v_proj']
            if use_sensitive:
                C4_grad_name_lookup_dict = {
                    linear_names[i]: C4_grad_name[i] for i in range(7)
                }

        else:
            raise NotImplementedError()
        
        # Insert sparse weights
        for i, layer in enumerate(transformer_layers):
            if use_sensitive:
                C4_grad_sq = C4_grad_sq_list[i]
            layer_modules = {n : module for n, module in layer.named_modules()}
            print(f"Inject sparse weight to layer {i}")

            for linear_name in linear_names:
                if use_sensitive:
                    C4_grad_sq_block = C4_grad_sq[C4_grad_name_lookup_dict[linear_name]]

                linear_module = layer_modules[linear_name]
                original_weight = linear_module.weight

                in_features = linear_module.weight.shape[1]
                out_features = linear_module.weight.shape[0]
                if use_sensitive:
                    d_plus_s_module = D_Plus_S_Linear(in_features, out_features, 
                                                    weight=original_weight,
                                                    bias=linear_module.bias, 
                                                    device=original_weight.device,
                                                    dtype=original_weight.dtype, 
                                                    C4_grad_sq_block=C4_grad_sq_block,
                                                    percentage=percentage, profiler=profiler)
                else:
                    d_plus_s_module = D_Plus_S_Linear(in_features, out_features, 
                                                  weight=original_weight,
                                                  bias=linear_module.bias, 
                                                  device=original_weight.device,
                                                  dtype=original_weight.dtype, 
                                                  C4_grad_sq_block=None,
                                                  percentage=percentage, profiler=profiler)
                d_plus_s_module.config = model.config
                layer_modules[linear_name] = d_plus_s_module
            
            if model.config.model_type == "opt":
                if need_all_linear:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.k_proj = layer_modules['self_attn.k_proj']
                    layer.self_attn.v_proj = layer_modules['self_attn.k_proj']
                    layer.self_attn.out_proj = layer_modules['self_attn.out_proj']
                    layer.fc1 = layer_modules['fc1']
                    layer.fc2 = layer_modules['fc2']

                else:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']

            elif model.config.model_type in ['llama', 'mistral']:
                if need_all_linear:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.k_proj = layer_modules['self_attn.k_proj']
                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']
                    layer.self_attn.o_proj = layer_modules['self_attn.o_proj']
                    layer.mlp.gate_proj = layer_modules['mlp.gate_proj']
                    layer.mlp.up_proj = layer_modules['mlp.up_proj']
                    layer.mlp.down_proj = layer_modules['mlp.down_proj']

                else:
                    layer.self_attn.q_proj = layer_modules['self_attn.q_proj']
                    layer.self_attn.v_proj = layer_modules['self_attn.v_proj']

            else:
                raise NotImplementedError()


        # Freeze non-sparse parameters
        for n, p in model.named_parameters():
            if "sparse" not in n:
                p.requires_grad = False


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


def attn_forward_hook(self, *args, **kwargs):
    """
    Replace the original attention forward with this to enable prefix
    """
    def _expand_bsz(x, bsz):
        if hasattr(self, 'num_key_value_heads'): # Mistral model
            x = x.reshape(x.size(0), self.num_key_value_heads, -1).transpose(0,1) 
        else:
            x = x.reshape(x.size(0), self.num_heads, -1).transpose(0,1) # (num_prefix, hidden) -> (num_head, num_prefix, hidden/num_head)
        x = x.unsqueeze(0).expand(bsz, *x.shape) # -> (bsz, num_head, num_prefix, hidden/num_head)
        return x
    
    if "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
    else:
        hidden_states = args[0]
    bsz = hidden_states.size(0)

    if 'past_key_value' not in kwargs or \
        kwargs['past_key_value'] is None or \
            (isinstance(kwargs['past_key_value'], transformers.cache_utils.DynamicCache) and \
              len(kwargs['past_key_value'].key_cache) <= self.layer_idx):
        if self.reparam:
            prefix_keys = self.prefix_mlp_keys(self.prefix_input_embeds)
            prefix_values = self.prefix_mlp_values(self.prefix_input_embeds)
        else:
            prefix_keys, prefix_values = self.prefix_keys, self.prefix_values
        
        if isinstance(kwargs['past_key_value'], transformers.cache_utils.DynamicCache):
            kwargs['past_key_value'].update(
                _expand_bsz(prefix_keys, bsz), 
                _expand_bsz(prefix_values, bsz), 
                self.layer_idx
            )
        else:
            kwargs['past_key_value'] = (_expand_bsz(prefix_keys, bsz), _expand_bsz(prefix_values, bsz))
    
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            am = kwargs['attention_mask']  
            kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1], self.num_prefix), dtype=am.dtype, device=am.device), am], dim=-1)
        elif len(args) > 1: # attention mask is passed via positional argument
            am = args[1]
            am = torch.cat([-torch.zeros((*am.shape[:-1], self.num_prefix), dtype=am.dtype, device=am.device), am], dim=-1)
            args = (args[0], am) + args[2:]

    return self.original_forward(*args, **kwargs)


def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    """
    Replace the original "prepare_inputs_for_generation" with this to pass prefix correctly
    """
    original_input_len = input_ids.size(-1)
    if past_key_values:
        input_ids = input_ids[:, -1:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    if past_key_values is not None:
        # Check if we should add extra to attention mask
        if past_key_values[0][0].size(2) != attention_mask.size(1) - 1:
            num_prefix = past_key_values[0][0].size(2) - (attention_mask.size(1) - 1)
            attention_mask = torch.cat([torch.ones((attention_mask.size(0), num_prefix), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=-1)

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


class PrefixTuning:

    def __init__(self, model, num_prefix, reparam=True, embed_dim=512, mid_dim=512, float16=False, init_by_real_act=False, profiler=None):
        """
        Inputs:
        num_prefix: number of prefix tokens
        reparam: use reparameterization trick (not used in MeZO)
        embed_dim, mid_dim: hyperparameters for reparameterization trick (not used in MeZO)
        float15: whether the model parameters are float15
        init_by_real_act: init prefix tokens by real activations
        """        
        self.model = model
        self.num_prefix = num_prefix 
        self.hidden_dim = model.config.hidden_size
        self.float16 = float16

        # Reparameterization 
        self.reparam = reparam
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.model.config.use_cache = True
        self.profiler = profiler

        input_embeds = None # For reparameterization
        if model.config.model_type == "opt":
            attention_name = "attn"
            first_layer_name = "layers.0"
            layer_name = "layers."
        elif model.config.model_type == "roberta":
            attention_name = "attention"
            first_layer_name = "layer.0"
            layer_name = "layer."
        elif model.config.model_type == "llama":
            attention_name = "attn"
            first_layer_name = "layers.0"
            layer_name = "layers."
        elif model.config.model_type == "mistral":
            attention_name = "attn"
            first_layer_name = "layers.0"
            layer_name = "layers."


        if init_by_real_act:
            # Initialize prefix with real words' activations
            assert not reparam

            # Randomly sample input tokens
            input_tokens = torch.randint(low=0, high=model.config.vocab_size, size=(1, num_prefix), dtype=torch.int64).cuda()
            with torch.inference_mode():
                # Get the real activations
                real_key_values = model(input_ids=input_tokens, use_cache=True).past_key_values


        # Insert prefix
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                layer_id = int(key.split(layer_name)[1].split(".")[0])
                print(f"Inject prefix to: {key}")
                _, _, attn = find_module(model, key)

                # Replace the old forward functions
                attn.original_forward = attn.forward
                attn.forward = attn_forward_hook.__get__(attn, type(attn))
                if not hasattr(attn, "num_heads"):
                    attn.num_heads = model.config.num_attention_heads
                first = first_layer_name in key
                self.add_prefix(attn, first=first, input_embeds=input_embeds)

                if first and self.reparam:
                    input_embeds = attn.prefix_input_embeds
                print(f"Reinitialize with actual activation: {key} (layer {layer_id})")
                keys = real_key_values[layer_id][0].squeeze(0).transpose(0, 1).reshape(num_prefix, -1)
                values = real_key_values[layer_id][1].squeeze(0).transpose(0, 1).reshape(num_prefix, -1)
                attn.prefix_keys.data = keys.to(attn.prefix_keys.data.device)
                attn.prefix_values.data = values.to(attn.prefix_values.data.device)

        # Freeze non-prefix parameters
        for n, p in model.named_parameters():
            if "prefix" not in n:
                p.requires_grad_(False)

        # Replace the old prepare_inputs_for_generation function 
        model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model, type(model))

    def add_prefix(self, module, first, input_embeds=None):
        device = module.k_proj.weight.data.device
        module.num_prefix = self.num_prefix
        module.reparam = self.reparam
        module.prefix_keys = nn.Parameter(torch.randn(self.num_prefix, self.hidden_dim, device=device, dtype=self.model.dtype), requires_grad=True)
        module.prefix_values = nn.Parameter(torch.randn(self.num_prefix, self.hidden_dim, device=device, dtype=self.model.dtype), requires_grad=True)

