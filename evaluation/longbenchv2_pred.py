import os
import json
import argparse
from tqdm import tqdm
import re
import numpy as np
import random
from specpv import Speculator, SpecConfig 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.multiprocessing as mp
import torch

def load_model_and_tokenizer(base_model_path, EAGLE_model_path, device):
    model = Speculator.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
        total_token=-1
    )
    model.eval()
    tokenizer = model.tokenizer
    return model, tokenizer


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def build_chat(tokenizer, text, model_name, cot=None):
    if 'llama3' in model_name:
        if cot is not None:
            messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": cot},
                {"role": "user", "content": "Based on the above, what is the single, most likely answer choice? Format your response as follows: 'The correct answer is (insert answer here)'."},
            ]
        else:
            messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    elif 'qwen3' in model_name:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        ) 
    return prompt

def clean_cot(text):
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    think_content = text
    if match:
        think_content = match.group(1)
        return f'<think>{think_content}</think>'
    else:
        match2 = re.search(r'<think>', text)
        if match2:
            return f'<think>{text[match2.end():]}</think>'
        else:
            return text

def get_longbench_v2_pred(data, args, out_path, spec_config, rank):
    model_name = os.path.basename(args.base_model).lower()
    device = torch.device(f'cuda:{rank}')
    template_0shot_cot = open(f'/home/lthpc/nvmessd/hcy/SpecPV/evaluation/config/prompts/{model_name}/0shot_cot_longbench_v2.txt', encoding='utf-8').read()
    
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.EAGLE_model, device)
    for item in tqdm(data, desc="processing"):
        context = item['context']
        template = template_0shot_cot
        
        prompt_text = template.replace(
            '$DOC$', context.strip()
        ).replace(
            '$Q$', item['question'].strip()
        ).replace(
            '$C_A$', item['choice_A'].strip()
        ).replace(
            '$C_B$', item['choice_B'].strip()
        ).replace(
            '$C_C$', item['choice_C'].strip()
        ).replace(
            '$C_D$', item['choice_D'].strip()
        )
        

        #generate cot
        prompt = build_chat(tokenizer, prompt_text, model_name)
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
        input_ids = model_inputs["input_ids"]
        if 'llama3' in model_name:
            output_ids ,metrics = model.spec_generate(input_ids,temperature=0,max_new_tokens=1024, max_length=65536, spec_config=spec_config, log=True,is_llama3=True)
        elif 'qwen3' in model_name:
            output_ids ,metrics = model.spec_generate(input_ids,temperature=0.6,top_p=0.95,top_k=20,max_new_tokens=1024, max_length=65536, spec_config=spec_config, log=True)

        avg_accept = metrics["avg_accept_length"]
        output_cot = model.tokenizer.decode(output_ids[0, input_ids.shape[1]:])
        output_think_cot = clean_cot(output_cot)
        item['response_cot'] = output_think_cot

        #generate answer
        if 'llama3' in model_name:
            prompt_ans = build_chat(tokenizer, prompt_text, model_name, output_cot)
            model_ans_inputs = tokenizer([prompt_ans], return_tensors="pt").to(device)
            input_ans_ids = model_ans_inputs["input_ids"]
            output_ans_ids, metrics = model.spec_generate(input_ans_ids,temperature=0,max_new_tokens=128, max_length=65536, spec_config=spec_config, log=True, is_llama3=True)

        elif 'qwen3' in model_name:
            output_q_ids = output_ids[0, :input_ids.shape[1]].unsqueeze(0).to(device)
            model_think_inputs =  tokenizer(output_think_cot, return_tensors="pt") .to(device)
            model_think_ids = model_think_inputs["input_ids"]
            input_ans_ids = torch.cat( [output_q_ids,model_think_ids],dim=1)
            output_ans_ids, metrics = model.spec_generate(input_ans_ids,temperature=0,max_new_tokens=128, max_length=65536, spec_config=spec_config, log=True)

        output = model.tokenizer.decode(output_ans_ids[0, input_ans_ids.shape[1]:])


        #print(output)
        item['avg_accept'] = avg_accept
        item['response'] = output.strip()
        item['pred'] = extract_answer(output)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        #print(item['judge'])
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def main():
    mp.set_start_method('spawn', force=True)
    os.makedirs(f"{args.save_dir}longbench_v2", exist_ok=True)

    model_name = os.path.basename(args.base_model).lower()    
    if not os.path.exists(f"outputs/{model_name}"):
        os.makedirs(f"outputs/{model_name}")
    out_file = f"outputs/{model_name}/longbench_v2/{args.method}.jsonl"
    if args.method == "specpv":
        out_file = f"outputs/{model_name}/longbench_v2/{args.method}-{args.partial_length}.jsonl"

    dataset = json.load(open("/home/lthpc/nvmessd/hcy/SpecPV/dataset/filtered_longbench_v2.json", 'r', encoding='utf-8'))


    data_all = [
        {
            "_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"], "length": item["length"], "question": item["question"],
            "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"],
            "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]
        } 
        for item in dataset
    ]
    
    has_data = set()
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                has_data.add(item["_id"])
    #fout = open(out_file, 'a', encoding='utf-8')
    data = [item for item in data_all if item["_id"] not in has_data]

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_longbench_v2_pred, args=(data_subsets[rank], args, out_file, spec_config, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print(f"Processing complete, results saved to: {out_file}")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", "-bm", type=str, default="/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct/")
    parser.add_argument("--EAGLE_model", "-em", type=str, default="/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K/")
    parser.add_argument('--method', type=str, default=None, choices=['specpv', 'full', 'naive'])
    parser.add_argument('--partial_length', type=str, default=None)
    parser.add_argument('--partial_spec_tokens', type=str, default='20')
    parser.add_argument("--offload", action="store_true",help="Enable offloading from GPU to CPU for KV cache or model weights.")
    parser.add_argument("--n_proc", "-n", type=int, default=4)
    args = parser.parse_args()
    
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
    if args.offload:
        spec_config.enable_offload = True
    print(spec_config)
    main()
