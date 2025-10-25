from specpv import Speculator
import torch

base_model_path = '/home/lab6033/zhendong/models/LLAMA3.1-8B-Instruct' 
EAGLE_model_path = '/home/lab6033/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B' 

# load model
model = Speculator.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()

vocab_size = 120000
for seqlen in [1024, 4096, 8192, 16384, 32768]:
    input_ids = torch.randint(low=0, high=vocab_size-1000, size=(1, seqlen), device="cuda")
    # warmup
    for _ in range(5):
        _ = model.spec_generate(input_ids, temperature=0.8, max_new_tokens=128, max_length=35000)
    output = model.spec_generate(input_ids,temperature=0.8, max_new_tokens=1024, max_length=35000)
    # get results
    from specpv.speculate.profile import print_time_stats,reset_time_stats
    print("########### seqlen:", seqlen, "###########")
    print_time_stats()
    reset_time_stats()


