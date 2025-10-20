from sparsesd import Speculator
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset

base_model_path = "/home/lab6033/zhendong/models/LLAMA3.1-8B-Instruct/"
#EAGLE_model_path = "/home/lab6033/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B/" 
EAGLE_model_path = "/home/lab6033/hcy/SparseSD/model/llama3-8b-eagle3-64k/"

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
# prepare input data
tokenizer = model.tokenizer


file_path = "/home/lthpc/nvmessd/hcy/SparseSD/data/pg-19/test_pg19_10k_to_60k.parquet"
df = pd.read_parquet(file_path)

for id in range(6,7):
   sum_accept = 0
   for i in range((id-1)*20,id*20):
      prompt = df['text'][i]

      system_prompt = "Please help me continue this story."
      messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": prompt}
      ]
      text = tokenizer.apply_chat_template(
         messages,
         tokenize=False,
         add_generation_prompt=True
      )
      
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model_inputs = tokenizer(text, return_tensors="pt").to(device)
      input_ids = model_inputs["input_ids"]
      
      #input_ids = model_inputs["input_ids"][:, : 1024]
      print("length:",len(input_ids[0]))
      # generate
      output_ids,accept_len=model.spec_generate(input_ids,temperature=0.7,max_new_tokens=128, max_length=10240*id+1000)
      #print(accept_len)
      #print(type(accept_len))
      #print(len(accept_len))
      avg_accept = sum(accept_len)/len(accept_len)
      sum_accept += avg_accept
   print(f"For {id}0k tokens, average accept length: {(sum_accept/20):.4f} tokens")
'''

prompt_format = "Answer the question based on the given passages.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages.\n\nQuestion: {input}\nLet's think step by step:"
file_path = "/home/lab6033/hcy/SparseSD/data/longbench-v1/2wikimqa_e.jsonl"

data = load_dataset("json", data_files=file_path, split="train")

acc_0_4k = []
acc_4_8k = []
acc_8_12k = []

for json_obj in data: 
   prompt = prompt_format.format(**json_obj)
   #print(prompt)
   messages=[{"role": "user", "content": prompt}]
   text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
   )
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model_inputs = tokenizer(text, return_tensors="pt").to(device)
   input_ids = model_inputs["input_ids"]
   length = len(input_ids[0])
   print("length:",length)
   if (length>12*1024):
      continue
   # generate
   output_ids,accept_len=model.spec_generate(input_ids,temperature=0.7,max_new_tokens=128, max_length=13000)
   #print(tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True))

   avg_accept = sum(accept_len)/len(accept_len)
   if (length<1024*4):
      acc_0_4k.append(avg_accept)
   elif (length<1024*8):
      acc_4_8k.append(avg_accept)
   elif (length<1024*12):
      acc_8_12k.append(avg_accept)
   print(avg_accept)
print(f"For 0-4k tokens, average accept length: {(sum(acc_0_4k)/len(acc_0_4k)):.4f} tokens")
print(f"For 4-8k tokens, average accept length: {(sum(acc_4_8k)/len(acc_4_8k)):.4f} tokens")
print(f"For 8-12k tokens, average accept length: {(sum(acc_8_12k)/len(acc_8_12k)):.4f} tokens")
'''