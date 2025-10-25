from datasets import load_dataset
from rouge import Rouge
from tqdm import tqdm
import numpy as np
from evaluate import load

# === file path ===
ground_truth_file = "/home/lab6033/zhendong/SpecPV/outputs/llama3.1-8b-instruct/qmsum-full.jsonl"
target_file = "/home/lab6033/zhendong/SpecPV/outputs/llama3.1-8b-instruct/qmsum-specpv-2048-20.jsonl"

gt_data = load_dataset("json", data_files=ground_truth_file)["train"]
tg_data = load_dataset("json", data_files=target_file)["train"]

# === map data ===
gt_dict = {item["answers"][0]: item["pred"].strip() for item in gt_data}
tg_dict = {item["answers"][0]: item["pred"].strip() for item in tg_data}

# === align samples ===
common_answers = sorted(set(gt_dict.keys()) & set(tg_dict.keys()))
if len(common_answers) < len(gt_dict):
    print(f"⚠️ Warning: Only match {len(common_answers)} / {len(gt_dict)} samples.")


refs, hyps = [], []
for ans in tqdm(common_answers, desc="Aligning samples"):
    refs.append(gt_dict[ans])
    hyps.append(tg_dict[ans])

# get rouge scores
rouge = Rouge()
rouge_scores = []

for pred, ref in tqdm(zip(hyps, refs), total=len(refs), desc="Computing ROUGE-L"):
    try:
        score = rouge.get_scores([pred], [ref], avg=True)["rouge-l"]["f"]
    except Exception:
        score = 0.0
    rouge_scores.append(score)

rouge_l_mean = np.mean(rouge_scores)
print(f"\nROUGE-L (mean F1): {rouge_l_mean:.4f}")

# get BLEURT
bleurt = load("bleurt", "bleurt-20")  
results = bleurt.compute(predictions=hyps, references=refs)

scores = results["scores"]
print(f"BLEURT (mean): {sum(scores)/len(scores):.4f}")