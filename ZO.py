import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple
from functools import partial
import math
import numpy as np
from transformers import default_data_collator
from torch.utils.data import DataLoader
from typing import List


@torch.no_grad()
def get_outlier_masks(model, percentile=0.005):
    masks = {n : None for n, p in model.named_parameters() if 'weight' in n}
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n: continue
        m = p.data.mean()
        top_cutoff = int(p.numel() * percentile)
        mask = torch.zeros(*p.shape, dtype=torch.bool, device=p.device)
        mask.view(-1)[(-((p - m).view(-1).abs())).argsort()[:top_cutoff]] = True
        masks[n] = mask
    return masks


@torch.no_grad()
def get_random_masks(model, percentile=0.005):
    masks = {n : None for n, p in model.named_parameters() if 'weight' in n}
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n: continue
        random_indices = torch.randperm(p.numel(), device=p.device)[:int(p.numel() * percentile)]
        mask = torch.zeros(*p.shape, dtype=torch.bool, device=p.device)
        mask.view(-1)[random_indices] = True
        masks[n] = mask
    return masks


# OBD approach:     
def optimal_brain_damage_masks(model: nn.Module, dataset,
                                  percentile=5e-3, microbatch=4, minibatch=64):
    assert minibatch % microbatch == 0
    count = 0
    train_dataloader = DataLoader(
        dataset, shuffle=True, collate_fn=default_data_collator, batch_size=microbatch)
    for sampled_batch in train_dataloader:
        count += len(sampled_batch['input_ids'])
        loss = model(**sampled_batch).loss
        loss /= (minibatch/microbatch)
        loss.backward()
        if count >= minibatch: 
            break
    
    mask_dict = dict()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if 'weight' not in n or 'layer_norm' in n: continue
            per_layer_scores = (p.grad ** 2)
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(*p.shape, dtype=torch.bool, device=p.device)
            mask.view(-1)[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
            mask_dict[n] = mask
            p.grad = None

    return mask_dict

# SqueezeLLM approach: gradient square as diagonal approx to Fisher info matrix
def gradient_square_masks(model: nn.Module, dataset,
                                  percentile=5e-3, microbatch=4, minibatch=64):
    assert minibatch % microbatch == 0
    count = 0
    train_dataloader = DataLoader(
        dataset, shuffle=True, collate_fn=default_data_collator, batch_size=microbatch)
    for sampled_batch in train_dataloader:
        count += len(sampled_batch['input_ids'])
        loss = model(**sampled_batch).loss
        loss /= (minibatch/microbatch)
        loss.backward()
        if count >= minibatch: 
            break
    
    mask_dict = dict()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if 'weight' not in n or 'layer_norm' in n: continue
            per_layer_scores = (p.grad ** 2)
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(*p.shape, dtype=torch.bool, device=p.device)
            mask.view(-1)[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
            mask_dict[n] = mask
            p.grad = None

    return mask_dict


# https://arxiv.org/pdf/1810.02340.pdf, SNIP approach
def get_gradient_weight_product_masks(model: nn.Module, dataset,
                                  percentile=5e-3, microbatch=1, minibatch=64):
    assert minibatch % microbatch == 0
    count = 0
    train_dataloader = DataLoader(
        dataset, shuffle=True, collate_fn=default_data_collator, batch_size=microbatch)
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n:
            p.requires_grad_(False)
    
    with torch.enable_grad():
        for sampled_batch in train_dataloader:
            sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}    
            count += len(sampled_batch['input_ids'])
            loss = model(**sampled_batch).loss
            loss /= (minibatch/microbatch)
            loss.backward()
            if count >= minibatch: 
                break
    
    mask_dict = dict()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            per_layer_scores = (p.grad.float() * p.data.float()).abs()
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(*p.shape, dtype=torch.bool, device=p.device)
            mask.view(-1)[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
            mask_dict[n] = mask
            p.grad = None

    return mask_dict


# https://arxiv.org/pdf/2002.07376.pdf, GraSP paper
def get_hessian_grad_weight_product_masks(model: nn.Module, dataset, 
                                  percentile=5e-3, microbatch=1, minibatch=64):
    assert minibatch % microbatch == 0
    count = 0
    for n, p in model.named_parameters():
        if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n:
            p.requires_grad_(False)
    
    train_dataloader = DataLoader(
        dataset, shuffle=True, collate_fn=default_data_collator, batch_size=microbatch)
    parameters_as_tuple = tuple(p for n, p in model.named_parameters() if p.requires_grad)
    grads = tuple(torch.zeros_like(p) for p in parameters_as_tuple)
    with torch.enable_grad():
        for sampled_batch in train_dataloader:  
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
            per_layer_scores = -(hessian_grad_prod[layer_count].float() * p.data.float())
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(*p.shape, dtype=torch.bool, device=p.device)
            mask.view(-1)[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
            mask_dict[n] = mask
            layer_count += 1

    return mask_dict



class ZOLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, ZO_training=True) -> None:
        super().__init__(in_features=in_features, 
                         out_features=out_features, 
                         bias=bias,
                         device=device, 
                         dtype=dtype)
        self.ZO_training = ZO_training

    # @torch.compile(mode='max-autotune')
    def ZO_forward_stack_weight_bias(self, X, W, W_eps, b, b_eps):
        h_stacked_w = torch.hstack([W + W_eps, W - W_eps])
        h_stacked_b = torch.hstack([b + b_eps, b - b_eps])
        return torch.baddbmm(h_stacked_b, X, h_stacked_w.permute(0, 1))

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.ZO_training:
            # assume input: [2B, L, dim]
            self.weight_eps = torch.randn_like(self.weight)
            if self.bias:
                self.bias_eps = torch.randn_like(self.bias)
                return self.ZO_forward_stack_weight_bias(
                    input, 
                    self.weight, 
                    self.weight_eps,
                    self.bias,
                    self.bias_eps
                )
            else:
                raise NotImplementedError()
                # self.bias_eps = None
                # return F.linear(input, self.weight, self.bias)
        else:
            # assume input: [B, L, dim]
            return super().forward(input)

    def eval_mode(self):
        pass

    def clear_cache(self):
        del self.bias_eps
        del self.weight_eps


@torch.no_grad()
def perturb_parameters(model: nn.Module, eps, one_trial_seed, scaling_factor=1):
    torch.manual_seed(one_trial_seed)

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data.add_(z, alpha=scaling_factor * eps)
    return model


@torch.no_grad()
def zo_perturb_parameters_with_mask(model: nn.Module, eps, one_trial_seed, mask_dict=None, scaling_factor=1):
    torch.manual_seed(one_trial_seed)
    
    for name, param in model.named_parameters():
        if name not in mask_dict: continue
        selected_param = param[mask_dict[name]]
        size = selected_param.size()
        z = torch.normal(mean=0, std=1, size=size, device=param.device, dtype=param.dtype)
        param[mask_dict[name]] += (scaling_factor * eps) * z

    return model


@torch.no_grad()
def zo_perturb_parameters_quantization(scale_param_list: List[torch.Tensor], 
                               eps, one_trial_seed, scaling_factor=1):
    torch.manual_seed(one_trial_seed)
    for s in scale_param_list:
        z = torch.normal(mean=0, std=1, size=s.size(), device=s.device, dtype=s.dtype)
        s.add_(z, alpha=scaling_factor * eps)


@torch.no_grad()
def zo_perturb_parameters_NF4(tunable_param_list: List[torch.Tensor], 
                               eps, one_trial_seed, scaling_factor=1):
    torch.manual_seed(one_trial_seed)
    for s in tunable_param_list:
        z = torch.normal(mean=0, std=1, size=s.size(), device=s.device, dtype=s.dtype)
        s.add_(z, alpha=scaling_factor * eps)


@torch.no_grad()
def project_soft_prompt_L2_min(soft_prompt: nn.Parameter, embed_token: nn.Parameter):
    # soft_prompt: P, d
    dist = torch.cdist(soft_prompt, embed_token)
    return embed_token[dist.argmin(dim=-1).indices]


@torch.no_grad()
def perturb_parameters_rademacher(model: nn.Module, eps, random_vector=None, scaling_factor=1):
    if random_vector is None:
        random_vector = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in random_vector:
            z = random_vector[name]
        else:
            z = torch.randint_like(param, high=2)
            z[z == 0] = -1
            random_vector[name] = z
        param.data.add_(z, alpha=scaling_factor * eps)
    return model, random_vector


@torch.no_grad()
def perturb_parameters_MeZO(model: nn.Module, eps, scaling_factor=1, seed=0):
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)    
        param.data.add_(z, alpha=scaling_factor * eps)
        del z
    return model


@torch.no_grad()
def perturb_parameters_one_column_LoRA(model: nn.Module, 
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


@torch.no_grad()
def perturb_parameters_one_layer_LoRA(model: nn.Module,  
                                       layer_dict, 
                                       random_vectors=None, 
                                       eps=-1,
                                       scaling_factor=-1):
    if random_vectors is None:
        random_vectors = dict()
    lora_params = { n : p for n, p in model.named_parameters() if 'lora' in n}
    for l, p in layer_dict.items():
        if l not in random_vectors:
            random_vectors[l] = torch.randn_like(p)
        lora_params[l].add_(random_vectors[l], alpha=scaling_factor*eps)
    return model, random_vectors


@torch.no_grad()
def copy_model_weight(model, params_dict):
    for n, p in model.parameters():
        if not p.requires_grad: continue
        p.data.copy_(params_dict[n])


@torch.no_grad()
def add_to_model(model, vector_scale_tuple):
    for vector, scale in vector_scale_tuple:
        if isinstance(vector, dict):
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                p.data.add_(vector[n], alpha=scale)
        else:
            for p in model.parameters():
                if not p.requires_grad: continue
                p.data.add_(vector, alpha=scale)


# def DFPI_SPSA(f, y, c_init, beta, T_power):
# 	# Power iteration - Compute eigenvector for max eigenvalue 
# 	r = 0.001
# 	T_power_approx = 5
# 	d2 = np.random.rand(f.d)

# 	c = c_init 
# 	for i in range(T_power_approx): 
# 		Delta = np.random.binomial(n=1, p=0.5, size=f.d)
# 		Delta[Delta == 0] = -1
# 		# Approximate gradient vectors 
# 		d_rplus = f.eval(y + r*d2 + c*Delta) - f.eval(y + r*d2 - c*Delta)
# 		G_rplus = np.divide(d_rplus, 2*c*Delta)

# 		d_rminus = f.eval(y - r*d2 + c*Delta) - f.eval(y - r*d2 - c*Delta)
# 		G_rminus = np.divide(d_rminus, 2*c*Delta)

# 		# Approximate Hessian-vector product
# 		Hd = (G_rplus - G_rminus)/(2*r)
		
# 		# Power iteration - update
# 		d2 = Hd/np.linalg.norm(Hd)

# 	# Approximate gradient vectors 
# 	d_rplus = f.eval(y + r*d2 + c*Delta) - f.eval(y + r*d2 - c*Delta)
# 	G_rplus = np.divide(d_rplus, 2*c*Delta)

# 	d_rminus = f.eval(y - r*d2 + c*Delta) - f.eval(y - r*d2 - c*Delta)
# 	G_rminus = np.divide(d_rminus, 2*c*Delta)

# 	# Approximate Hessian-vector product
# 	Hd = (G_rplus - G_rminus)/(2*r)

# 	# Largest eigenvalue 
# 	lmax = np.linalg.norm(Hd)/np.linalg.norm(d2)

# 	# Power iteration - Compute eigenvector for min eigenvalue 
# 	b_power = 1/lmax
# 	d2 = np.random.rand(f.d)
# 	for i in range(T_power): 
# 		Delta = np.random.binomial(n=1, p=0.5, size=f.d)
# 		Delta[Delta == 0] = -1

# 		# Approximate gradient vectors 
# 		d_rplus = f.eval(y + r*d2 + c*Delta) - f.eval(y + r*d2 - c*Delta)
# 		G_rplus = np.divide(d_rplus, 2*c*Delta)

# 		d_rminus = f.eval(y - r*d2 + c*Delta) - f.eval(y - r*d2 - c*Delta)
# 		G_rminus = np.divide(d_rminus, 2*c*Delta)

# 		# Approximate Hessian-vector product
# 		Hd = (G_rplus - G_rminus)/(2*r)
		
# 		# Power iteration - update
# 		d2_ = d2 - b_power*Hd
# 		d2  = d2_/np.linalg.norm(d2_)

# 	# Negative curvature 
# 	return d2


@torch.no_grad()
def Rademacher(model, eps=1):
    random_vector = {name: eps * torch.randint(0, 1, param.size(), dtype=param.dtype, device=param.device) 
                     for name, param in model.named_parameters() if param.requires_grad}
    for v in random_vector.values():
        v[v == 0] = -eps
    return random_vector


@torch.no_grad()
def compute_ZO_Hessian_eigenvector(model, eps, batch, power_iter=5):
    random_vector = {n: torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype) 
                     for n, p in model.named_parameters() if p.requires_grad}
    for _ in range(power_iter):
        rad = Rademacher(model, 1)
        add_to_model(model, [(random_vector, eps), (rad, eps)])
        loss1_1 = model(batch).loss

        add_to_model(model, [(rad, -2 * eps)])
        loss2_1 = model(batch).loss

        grad_plus = {n : (loss1_1 - loss2_1) / (2 * eps * z) for n, z in random_vector.items()}

        add_to_model(model, [(random_vector, -2 * eps), (rad, 2 * eps)])
        loss1_2 = model(batch).loss

        add_to_model(model, [(rad, -2 * eps)])
        loss2_2 = model(batch).loss

        add_to_model(model, [(random_vector, eps), (rad, eps)])

        grad_minus = {n : (loss1_2 - loss2_2) / (2 * eps * z) for n, z in random_vector.items()}

        Hd = {n : (g_plus - grad_minus[n]) / (2 * eps) for n, g_plus in grad_plus.items()}
        random_vector = {n: hd / torch.norm(hd, p=2) for n, hd in Hd.items()}
    
    b_power = {n: torch.norm(hd) / torch.norm(random_vector[n]) for n, hd in Hd.items()}
    random_vector = {n: torch.normal(mean=0, std=1, size=p.size(), device=p.device, dtype=p.dtype) 
                     for n, p in model.named_parameters() if p.requires_grad}

    for _ in range(power_iter):
        rad = Rademacher(model, 1)
        add_to_model(model, [(random_vector, eps), (rad, eps)])
        loss1_1 = model(batch).loss

        add_to_model(model, [(rad, -2 * eps)])
        loss2_1 = model(batch).loss

        grad_plus = {n: (loss1_1 - loss2_1) / (2 * eps * z)
                     for n, z in random_vector.items()}

        add_to_model(model, [(random_vector, -2 * eps), (rad, 2 * eps)])
        loss1_2 = model(batch).loss

        add_to_model(model, [(rad, -2 * eps)])
        loss2_2 = model(batch).loss

        add_to_model(model, [(random_vector, eps), (rad, eps)])

        grad_minus = {n: (loss1_2 - loss2_2) / (2 * eps * z)
                      for n, z in random_vector.items()}

        Hd = {n: (g_plus - grad_minus[n]) / (2 * eps) for n, g_plus in grad_plus.items()}
        
        random_vector = {n: random_vector[n] - b_power[n] * Hd[n] for n, g_plus in grad_plus.items()}
        random_vector = {n: d / torch.norm(d) for n, d in random_vector.items()}

    return random_vector



@torch.no_grad()
def JVP_forward_gradients(partial_loss_func, 
                          params: Tuple[torch.Tensor],
                          input_ids, 
                          attention_mask, 
                          labels):
    v_params = tuple([torch.randn_like(p) for p in params])

    loss_func = partial(
        partial_loss_func,
        input_ids, 
        attention_mask, 
        labels
    )
    # Forward AD
    loss, jvp = torch.func.jvp(loss_func, (params,), (v_params,))

    # Setting gradients
    for v, p in zip(v_params, params):
        p.grad = v * jvp
    
    return loss


class InverseSquareRootEpsilonSchedule:
    def __init__(self, init_eps, smoothing=100) -> None:
        self.init_eps = init_eps
        self.iter = 0
        self.smoothing = smoothing
    
    def step(self):
        self.iter += 1
        return max(1e-4, min(1, 1 / math.sqrt(self.iter / self.smoothing)) * self.init_eps)


class LinearEpsilonSchedule:
    def __init__(self, init_eps, total_steps) -> None:
        self.init_eps = init_eps
        self.iter = 0
        self.total_steps = total_steps
    
    def step(self):
        self.iter += 1
        return max(5e-4, self.init_eps * (self.total_steps - self.iter) / self.total_steps)


class CosineEpsilonSchedule:
    def __init__(self, init_eps, total_steps, num_cycles=0.5) -> None:
        self.init_eps = init_eps
        self.iter = 0
        self.num_cycles = num_cycles
        self.total_steps = total_steps

    def step(self):
        self.iter += 1
        progress = self.iter / self.total_steps
        return max(5e-4, 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress)) * self.init_eps)


class ConstantEpsilonSchedule:
    def __init__(self, init_eps) -> None:
        self.init_eps = init_eps

    def step(self):
        return self.init_eps


class StepEpsilonSchedule:
    def __init__(self, init_eps, total_steps) -> None:
        self.init_eps = init_eps
        self.total_steps = total_steps
        self.iter = 0

    def step(self):
        self.iter += 1
        if self.iter < self.total_steps / 3:
            return self.init_eps
        elif self.iter > self.total_steps / 3 and self.iter < 2 * self.total_steps / 3:
            return self.init_eps * 0.1
        else:
            return max(self.init_eps * 0.01, 5e-4)


class ConstantRatioEpsilonSchedule:
    def __init__(self, eps, model, mu=0.999) -> None:
        self.eps = eps
        self.model = model
        self.w_2 = math.sqrt(sum(torch.sum(p ** 2).cpu().item()
                             for p in model.parameters() if p.requires_grad))
        self.mu = mu

    @torch.no_grad()
    def step(self):
        w_2 = math.sqrt(sum(torch.sum(p ** 2).cpu().item() 
                            for p in self.model.parameters() if p.requires_grad))        
        eps = self.w_2 / w_2 * self.eps 
        self.w_2 = self.mu * self.w_2 + (1 - self.mu) * w_2
        return eps


class StepZOSampleSchedule:
    def __init__(self, total_steps) -> None:
        self.total_steps = total_steps
        self.iter = 0

    def step(self):
        self.iter += 1
        if self.iter < self.total_steps / 4:
            return 1
        elif self.iter > self.total_steps / 4 and self.iter < self.total_steps / 2:
            return 2
        else:
            return 3

@torch.no_grad()
def LoRA_more_rank(A, rank_total, U_rand=True, V_rand=False):
    # A: prompt_token, embed_dim
    prompt_token, embed_dim = A.shape
    U, S, V = torch.svd(A)
    V = V.T
    for i in range(len(S)):
        U[:, i] *= math.sqrt(S[i])
        V[i, :] *= math.sqrt(S[i])

    if rank_total <= prompt_token:
        return U[:, :rank_total], V[:rank_total, :]
    else:
        ret_U = torch.zeros(prompt_token, rank_total, device=A.device, dtype=A.dtype)
        ret_U[:, :prompt_token] = U
        if U_rand:
            ret_U[:, prompt_token:] = torch.randn_like(ret_U[:, prompt_token:])

        ret_V = torch.zeros(rank_total, embed_dim, device=A.device, dtype=A.dtype)
        ret_V[:prompt_token, :] = V
        if V_rand:
            ret_V[prompt_token:, :] = torch.randn_like(ret_V[prompt_token:, :])

        return ret_U, ret_V


@torch.no_grad()
def LoRA_more_rank_2_way_LoRA(A, rank_total, U_rand=True, V_rand=False):
    # A: prompt_token, embed_dim
    embed_dim, interm_rank = A.shape
    U, S, V = torch.svd(A)
    V = V.T

    if rank_total <= interm_rank:
        return U[:, :rank_total], V[:rank_total, :]
    else:
        ret_U = torch.zeros(embed_dim, rank_total,
                            device=A.device, dtype=A.dtype)
        ret_U[:, :interm_rank] = U
        if U_rand:
            ret_U[:, interm_rank:] = torch.randn_like(ret_U[:, interm_rank:])

        ret_V = torch.zeros(rank_total, interm_rank,
                            device=A.device, dtype=A.dtype)
        ret_V[:interm_rank, :] = V
        if V_rand:
            ret_V[interm_rank:, :] = torch.randn_like(ret_V[interm_rank:, :])

        return ret_U, ret_V


@torch.no_grad()
def LoRA_more_rank_qr(A, rank_total, U_rand=True, V_rand=False):
    # A: prompt_token, embed_dim
    prompt_token, embed_dim = A.shape
    Q, R = torch.linalg.qr(A, mode='complete')

    ret_Q = torch.zeros(prompt_token, rank_total, device=A.device, dtype=A.dtype)
    ret_Q[:, :prompt_token] = Q
    if U_rand:
        ret_Q[:, prompt_token:] = torch.randn_like(ret_Q[:, prompt_token:])

    ret_R = torch.zeros(rank_total, embed_dim, device=A.device, dtype=A.dtype)
    ret_R[:prompt_token, :] = R
    if V_rand:
        ret_R[prompt_token:, :] = torch.randn_like(ret_R[prompt_token:, :])

    return ret_Q, ret_R


@torch.no_grad()
def LoRA_more_rank_random_initialized(A, rank_total):
    # A: prompt_token, embed_dim
    prompt_token, embed_dim = A.shape
    U = torch.randn(prompt_token, rank_total, device=A.device, dtype=A.dtype) * 2 / math.sqrt(rank_total)
    V = torch.linalg.pinv(U) @ A
    return U, V


@torch.no_grad()
def random_orthonormal(dim, rank_total, device):
    H = torch.randn(dim, dim, device=device)
    U, S, Vt = torch.linalg.svd(H)
    return U[:rank_total]


@torch.no_grad()
def LoRA_more_rank_random_orthonormal_initialized(A, rank_total):
    prompt_token, embed_dim = A.shape
    U = random_orthonormal(rank_total, prompt_token, device=A.device)
    V = torch.linalg.pinv(U) @ A
    return U, V

# @torch.no_grad()
# def random_orthonormal(dim, rank_total, device):
#     eye = torch.eye(dim, device=device)[torch.from_numpy(
#         np.random.permutation(dim)[:rank_total])]
#     return eye + torch.randn_like(eye) * 1e-4


def LoRA_more_rank_normalized(A, rank_total):
    # A: prompt_token, embed_dim
    prompt_token, embed_dim = A.shape
    U, S, V = torch.svd(A)
    V = V.T
    for i in range(len(S)):
        U[:, i] *= math.sqrt(S[i])
        V[i, :] *= math.sqrt(S[i])

    ret_U = torch.zeros(prompt_token, rank_total, device=A.device, dtype=A.dtype)
    ret_U[:, :prompt_token] = U
    ret_U[:, prompt_token:] = torch.randn_like(ret_U[:, prompt_token:]) / (rank_total - prompt_token)

    ret_V = torch.zeros(rank_total, embed_dim, device=A.device, dtype=A.dtype)
    ret_V[:prompt_token, :] = V

    return ret_U, ret_V


def LoRA_more_rank_normalized(A, rank_total):
    # A: prompt_token, embed_dim
    prompt_token, embed_dim = A.shape
    U, S, V = torch.svd(A)
    V = V.T
    for i in range(len(S)):
        U[:, i] *= math.sqrt(S[i])
        V[i, :] *= math.sqrt(S[i])

    ret_U = torch.zeros(prompt_token, rank_total, device=A.device, dtype=A.dtype)
    ret_U[:, :prompt_token] = U
    ret_U[:, prompt_token:] = torch.randn_like(ret_U[:, prompt_token:])
    ret_U[:, prompt_token:] /= torch.norm(ret_U[:, prompt_token:], dim=0)

    ret_V = torch.zeros(rank_total, embed_dim, device=A.device, dtype=A.dtype)
    ret_V[:prompt_token, :] = V

    return ret_U, ret_V


def LoRA_more_rank_two_zeros(A, rank_total):
    # A: prompt_token, embed_dim
    prompt_token, embed_dim = A.shape
    U, S, V = torch.svd(A)
    V = V.T
    for i in range(len(S)):
        U[:, i] *= math.sqrt(S[i])
        V[i, :] *= math.sqrt(S[i])

    ret_U = torch.zeros(prompt_token, rank_total, device=A.device, dtype=A.dtype)
    ret_U[:, :prompt_token] = U

    ret_V = torch.zeros(rank_total, embed_dim, device=A.device, dtype=A.dtype)
    ret_V[:prompt_token, :] = V

    return ret_U, ret_V



# def perturb_single_layer(model, layer_name, eps, random_vector=None, scaling_factor=1):
#     if random_vector is None:
#         random_vector = {}

#     for name, param in self.named_parameters_to_optim:
#         cname = self.retrieve_c(name)
#         if cname == layer_name:
#             if name in random_vector:
#                 z = random_vector[name]
#             else:
#                 z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
#                 random_vector[name] = z
#             param.data = param.data + scaling_factor * z * eps

#     return model, random_vector


# def initialize_c(self, model, inputs):
#     self.named_parameters_to_optim = []
#     for name, param in model.named_parameters():
#         if self.should_optim(name, param):
#             self.named_parameters_to_optim.append((name, param))

#     self.cs = {'embed': 0.0, 'lm_head': 0.0} 
#     # OPT: embed_tokens; embed_positions
#     # RoBERTa: embeddings
#     self.num_params = copy.deepcopy(self.cs)
#     self.num_model_layers = model.config.num_hidden_layers
#     layer_name = "layers" if model.config.model_type == "opt" else "layer"
#     for i in range(self.num_model_layers): 
#         self.cs[f'{layer_name}.{i}.'] = 0.0
#         self.num_params[f'{layer_name}.{i}.'] = 0
    
#     # ZO estimation of c's
#     if self.args.zo_variant != 'param_norm' and self.args.use_zo_grad_est:
#         for layer in self.cs.keys():
#             with torch.no_grad():
#                 model, z = self.perturb_single_layer(model, layer_name=layer)
#                 loss1 = self.zo_forward(model, inputs)
#                 model, z = self.perturb_single_layer(model, layer_name=layer, random_vector=z, scaling_factor=-2)
#                 loss2 = self.zo_forward(model, inputs)

#             projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)
#             self.cs[layer] = torch.abs(projected_grad)

#             model, z = self.perturb_single_layer(model, layer_name=layer, random_vector=z)
    
#     # no need to run backprop if we are using parameter norm variant, can just measure them
#     elif self.args.zo_variant == 'param_norm':
#         for name, param in self.named_parameters_to_optim:
#             print(name)
#             ckey = self.retrieve_c(name)
#             if ckey in self.cs:
#                 self.cs[ckey] += torch.sum(param.data ** 2)
#                 self.num_params[ckey] += param.data.numel()

#         # take sqrt to get norm
#         for ckey in self.cs:
#             self.cs[ckey] = torch.sqrt(self.cs[ckey])
#             if self.args.scale_norm_by_num_params:
#                 self.cs[ckey] /= torch.sqrt(self.cs[ckey])
        
#         for ckey in self.cs:
#             if self.cs[ckey] != 0:
#                 self.cs[ckey] = self.cs[ckey].detach().item()
    
#     # backpropagation estimation fo ZO c's
#     #   this is mostly for debugging purposes to disentangle the variance from using ZO to estimate c
#     #   from the effectiveness of the preconditioners
#     else: 
#         model.eval()
#         inputs = self._prepare_inputs(inputs)
#         with self.compute_loss_context_manager():
#             loss = self.compute_loss(model, inputs)
#         if self.args.n_gpu > 1:
#             loss = loss.mean()  # mean() to average on multi-gpu parallel training
#         loss.backward()
#         for name, param in self.named_parameters_to_optim:
#             if param.grad is None:
#                 print(name)
#             else:
#                 ckey = self.retrieve_c(name)
#                 if ckey in self.cs:
#                     self.cs[ckey] += torch.sum(param.grad ** 2)
#                     self.num_params[ckey] += param.grad.numel()

#         # take sqrt to get norm
#         for ckey in self.cs:
#             self.cs[ckey] = torch.sqrt(self.cs[ckey])
#             if self.args.scale_norm_by_num_params:
#                 self.cs[ckey] /= torch.sqrt(self.num_params[ckey])

#         for ckey in self.cs:
#             if self.cs[ckey] != 0:
#                 self.cs[ckey] = self.cs[ckey].detach().item()

#     self.layer_names = list(self.cs.keys())
#     model.zero_grad()


# def retrieve_c(self, param_name):
#     for c_name in self.cs.keys():
#         if c_name in param_name:
#             return c_name
#     return '' # these parameters are likely not being used in the forward pass
