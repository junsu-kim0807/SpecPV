from sparsesd import Speculator
import torch

base_model_path = '/home/lab6033/zhendong/models/LLAMA3.1-8B-Instruct' 
EAGLE_model_path = '/home/lab6033/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B' 

# load model
model = Speculator.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()
# prepare input data
tokenizer = model.tokenizer
prompt = "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?"
system_prompt = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
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
model_inputs = tokenizer([text], return_tensors="pt").to(device)
input_ids = model_inputs["input_ids"]

# generate
output_ids=model.ea_generate(input_ids,temperature=0.8,max_new_tokens=1024)
output=model.tokenizer.decode(output_ids[0])
print(output)