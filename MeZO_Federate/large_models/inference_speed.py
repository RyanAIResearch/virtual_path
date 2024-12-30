import torch
import warnings
warnings.filterwarnings('ignore')
import eventProfiler
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import argparse
import os


@torch.inference_mode()
def fixed_seq_CLM(model, profiler, L_list, ITER, skip_count, record_time_string, record_total_time_instead=False):
    
    final_ret = dict()
    for L in L_list:
        for _ in tqdm(range(ITER)):
            X = torch.randint(2000, 30000, (16, L,)).cuda()
            if record_total_time_instead:
                with profiler('total time'):
                    model(input_ids=X)
            else:
                model(input_ids=X)
        
        ret = dict()
        if record_total_time_instead:
            res = profiler.summary()['time']['total time']
            res = res[skip_count:]
            res = res * 1000
            print(f'total time, L, {res.mean().item():.2f}, {res.std().item():.2f}')
            ret['total time'] = res

        else:
            for s in record_time_string:
                res = profiler.summary()['time'][s]
                call_per_forward = len(res) // ITER
                res = res[skip_count * call_per_forward:]
                res = res.reshape(len(res)//call_per_forward, call_per_forward)
                res = res * 1000
                res = res.sum(-1)
                print(f'{s}, L, {res.mean().item():.2f}, {res.std().item():.2f}')
                ret[s] = res

        profiler.reset()
        final_ret[L] = ret
    return final_ret


@torch.inference_mode()
def generation(model, tokenizer, profiler, max_length_list, ITER, skip_count, record_time_string, record_total_time_instead=False):
    prompt = "Please describe the effect of sparse zeroth-order optimization methods on memory-efficient LLM fine-tuning: "
    prompt_ids = tokenizer([prompt], return_tensors='pt')['input_ids']
    prompt_ids = prompt_ids.cuda()

    final_ret = dict()
    for max_length in max_length_list:
        for _ in tqdm(range(ITER)):
            if record_total_time_instead:
                with profiler('total time'):
                    model.generate(
                        prompt_ids, do_sample=True, 
                        top_p=0.95,
                        max_new_tokens=max_length, 
                        min_new_tokens=max_length, use_cache=True 
                    )    
            else:
                model.generate(
                    prompt_ids, do_sample=True, 
                    top_p=0.95,
                    max_new_tokens=max_length, 
                    min_new_tokens=max_length, use_cache=True 
                )

        ret = dict()
        if record_total_time_instead:
            res = profiler.summary()['time']['total time']
            res = res[skip_count:]
            res = res * 1000
            print(f'total time, L, {res.mean().item():.2f}, {res.std().item():.2f}')
            ret['total time'] = res

        else:
            for s in record_time_string:
                res = profiler.summary()['time'][s]
                call_per_forward = len(res) // ITER
                res = res[skip_count * call_per_forward:]
                res = res.reshape(len(res)//call_per_forward, call_per_forward)
                res = res * 1000
                res = res.sum(-1)
                print(f'{s}, L, {res.mean().item():.2f}, {res.std().item():.2f}')
                ret[s] = res

        profiler.reset()
        final_ret[max_length] = ret
    return final_ret


parser = argparse.ArgumentParser(
    description="test inference speed")
parser.add_argument(
    "--type",
    type=str,
    default='lora',
    choices=['lora', 'sensitive_sparse', 'random_sparse', 'prefix', 'vanilla']
)
parser.add_argument(
    "--model_name",
    type=str,
    default='NousResearch/Llama-2-7b-hf',
)
parser.add_argument(
    "--C4_grad_addr",
    type=str,
    default='/share/desa/nfs02/wg247/SqueezeLLM/grad/llama2-7b-chunk',
)
parser.add_argument(
    "--sparse_percentage_list",
    type=str,
    default='1e-4,1e-3,1e-2,1e-1'
)
parser.add_argument(
    "--lora_rs",
    type=str,
    default='4,8,16'
)
parser.add_argument(
    "--prefix_lengths",
    type=str,
    default='5,10,20'
)
args = parser.parse_args()
profiler = eventProfiler.EventProfiler(torch.device('cuda:0'), True, True)


L0 = 1024
L1 = 128

with torch.inference_mode():
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    LAYERS = 32

    if args.type == 'lora':
        from inference_test import LoRA
        final_results = dict()
        for r in args.lora_rs.split(','):
            r = int(r)

            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map='auto',
                attn_implementation='flash_attention_2',
                torch_dtype=torch.float16
            )
            model.eval()
            LoRA(model, r=r, alpha=16, profiler=profiler)

            trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
            print(f'trainable params {r} {trainable_params}')

            clm_ret = fixed_seq_CLM(model, profiler, L0, 25, 5, ['lora'])
            gen_ret = generation(model, tokenizer, profiler, L1, 5, 0, ['lora'])
            final_results[r] = (
                clm_ret, gen_ret, trainable_params
            )

            del model
            torch.cuda.empty_cache()

        print()
        print(final_results)
        torch.save(final_results, 'lora-inference-time.pt')

    elif args.type in ['sensitive_sparse', 'random_sparse']:
        from inference_test import SparseTuning

        if args.type == 'sensitive_sparse':
            C4_grad_sq_list = []
            for i in range(LAYERS):
                one_layer = torch.load(args.C4_grad_addr + os.sep + f'layer_{i}.pt', map_location='cuda:0') 
                one_layer = {k : v.square() for k, v in one_layer.items()}
                C4_grad_sq_list.append(one_layer)
            
        final_results = dict()
        for sparse_percentage in args.sparse_percentage_list.split(','):
            sparse_percentage = float(sparse_percentage)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map='auto',
                attn_implementation='flash_attention_2',
                torch_dtype=torch.float16
            )
            model.eval()
            if args.type == 'sensitive_sparse':
                SparseTuning(model, sparse_percentage, C4_grad_sq_list, profiler, use_sensitive=True)
            elif args.type == 'random_sparse':
                SparseTuning(model, sparse_percentage, None, profiler, use_sensitive=False)
            else:
                raise NotImplementedError()

            trainable_params = sum(p.values().numel() for n, p in model.named_parameters() if p.requires_grad)
            print(f'trainable params {sparse_percentage} {trainable_params}')

            clm_ret = fixed_seq_CLM(model, profiler, [256, 512, 1024], 15, 5, ['sparse-add', 'sparse-mm'])
            # model.config.disable_sparse_mm = True
            gen_ret = generation(model, tokenizer, profiler, [128], 3, 0, ['sparse-add', 'sparse-mm'])

            final_results[sparse_percentage] = (
                clm_ret, gen_ret, trainable_params
            )
            del model
            torch.cuda.empty_cache()
        
        print()
        print(final_results)
        if args.type == 'sensitive_sparse':
            torch.save(final_results, 'sensitive_sparse-inference-time.pt')
        elif args.type == 'random_sparse':
            torch.save(final_results, 'random_sparse-inference-time.pt')
        else:
            raise NotImplementedError()

    elif args.type == 'prefix': # prefix
        from inference_test import PrefixTuning

        final_results = dict()
        for length in args.prefix_lengths.split(','):
            length = int(length)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map='auto',
                attn_implementation='flash_attention_2',
                torch_dtype=torch.float16
            )
            model.eval()
            PrefixTuning(model, num_prefix=length, reparam=False, float16=True, init_by_real_act=True)

            trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
            print(f'trainable params {length} {trainable_params}')

            clm_ret = fixed_seq_CLM(model, profiler, L0, 25, 5, [], record_total_time_instead=True)
            gen_ret = generation(model, tokenizer, profiler, L1, 10, 0, [], record_total_time_instead=True)

            final_results[length] = (
                clm_ret, gen_ret, trainable_params
            )
            del model
            torch.cuda.empty_cache()
        
        print()
        print(final_results)
        torch.save(final_results, 'prefix-inference-time.pt')

    elif args.type == 'vanilla':
        final_results = dict()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map='auto',
            attn_implementation='flash_attention_2',
            torch_dtype=torch.float16
        )
        model.eval()
            
        clm_ret = fixed_seq_CLM(model, profiler, L0, 25, 5, [], record_total_time_instead=True)
        gen_ret = generation(model, tokenizer, profiler, L1, 10, 0, [], record_total_time_instead=True)

        final_results['vanilla'] = (
            clm_ret, gen_ret
        )

        print()
        print(final_results)
        torch.save(final_results, 'vanilla-inference-time.pt')
