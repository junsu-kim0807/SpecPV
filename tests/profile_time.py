from datasets import load_dataset
from specpv import Speculator, SpecConfig
import torch
from specpv.speculate.profile import print_time_stats,reset_time_stats

# Prepare dataset
dataset = load_dataset("parquet", data_files="data/pg19_test/pg19_test_60k.parquet")['train']
text = dataset[0]['text']
base_model_path = "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct"
EAGLE_model_path = "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K"
# load model
model = Speculator.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()

tokenizer = model.tokenizer
system_prompt = (
    "You are a creative writing assistant. Continue the following story "
    "in a coherent, engaging, and stylistically consistent way."
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": text}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_inputs = tokenizer([text], return_tensors="pt").to(device)
input_ids = model_inputs["input_ids"]
# generate
seq_lens = [2048, 4096, 8192, 16384, 32768, 65536]
spec_config = SpecConfig(enable_offload=False, enable_partial_kv=False, n_retrieval_blocks=512, partial_spec_tokens=20)

# warm up 
for _ in range(3):
    output_ids,metrics=model.spec_generate(input_ids[:, :20000],temperature=0,max_new_tokens=64, max_length=65000, log=True, is_llama3=True, spec_config=spec_config)

# start record
for seqlen in seq_lens:
    print(f"\n==============================")
    print(f"### Testing sequence length: {seqlen}")
    print(f"==============================")

    reset_time_stats()

    for idx in range(5): 
        text = dataset[idx]['text']
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        input_ids = model_inputs["input_ids"]

        # truncate to seqlen
        trunc_len = min(seqlen, input_ids.shape[-1])
        input_ids = input_ids[:, :trunc_len]

        output_ids, metrics = model.spec_generate(
            input_ids,
            temperature=0,
            max_new_tokens=512,
            max_length=65000,
            log=True,
            is_llama3=True,
            spec_config=spec_config
        )

        print(f"Sample {idx} avg_accept_length = {metrics['avg_accept_length']}")

    print_time_stats()


