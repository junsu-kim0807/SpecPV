import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from specpv import Speculator, SpecConfig 


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--draft_path', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None, choices=['gov_report', 'qmsum', 'multi_news', 'all'])
    parser.add_argument('--method', type=str, default=None, choices=['specpv', 'full', 'naive'])
    parser.add_argument('--partial_length', type=str, default=None)
    parser.add_argument('--partial_spec_tokens', type=str, default='20')
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, data, max_gen, prompt_format, dataset, device, model_path, draft_path, out_path, model_name, spec_config, method):
    device = torch.device(f'cuda:{rank}')
    max_length = 60000
    model, tokenizer = load_model_and_tokenizer(model_path, draft_path, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer([prompt], return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        is_llama3 = True if "llama3" in model_name else False
        if method == "naive":
            output, metrics = model.naive_generate(
                input_ids=input.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length+max_gen+100,
                temperature=0.0,
                is_llama3=is_llama3,
                log=True
            )
            metrics["avg_accept_length"] = 0  # dummy value
        else:
            output, metrics = model.spec_generate(
                input_ids=input.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length+max_gen+100,
                temperature=0.0,
                is_llama3=is_llama3,
                spec_config=spec_config,
                log=True
            )
        pred = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], 
                       "all_classes": json_obj["all_classes"], 
                       "length": context_length, 
                       "avg_acc_length": metrics['avg_accept_length'], 
                       "new_token": metrics["new_token"], 
                       "total_time": metrics["total_time"],
                       "throughput": metrics["throughput"]
                       }, f, ensure_ascii=False)
            f.write('\n')
    if dist.is_initialized():
        dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_path, draft_path, device):
    model = Speculator.from_pretrained(
        base_model_path=model_path,
        ea_model_path=draft_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
        total_token=-1
    )
    model.eval()
    tokenizer = model.tokenizer
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset_name == 'all':
        datasets = ["gov_report", "qmsum", "multi_news"]
    else:
        datasets = [args.dataset_name]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("evaluation/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("evaluation/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # set spec config
    spec_config = SpecConfig()
    if args.method == "full":
        spec_config.enable_partial_kv = False
    elif args.method == "specpv":
        spec_config.partial_spec_tokens = int(args.partial_spec_tokens)
        spec_config.enable_partial_kv = True
        spec_config.n_retrieval_blocks = int(args.partial_length ) // spec_config.block_size
    else:
        print("Naive generation, no spec config needed.")
    print(spec_config)
    model_name = os.path.basename(args.model_path).lower()
    for dataset in datasets:
        # data = load_dataset('THUDM/LongBench', dataset, split='test')
        data = load_dataset('json', data_files=f'data/longbenchv1/{dataset}.jsonl')['train']
        # data = data.select(range(5))
        if not os.path.exists(f"outputs/{model_name}"):
            os.makedirs(f"outputs/{model_name}")
        out_path = f"outputs/{model_name}/{dataset}-{args.method}.jsonl"
        if args.method == "specpv":
            out_path = f"outputs/{model_name}/{dataset}-{args.method}-{args.partial_length}-{args.partial_spec_tokens}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, data_subsets[rank], \
                        max_gen, prompt_format, dataset, device, args.model_path, args.draft_path, out_path, model_name, spec_config, args.method))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
