from datasets import load_dataset
import numpy as np
import os

def process_outputs(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = load_dataset("json", data_files=file_path)["train"]
    print(f"\n=== Analyzing {os.path.basename(file_path)} ===")

    # ---- 1. Throughput ----
    if all(k in data.column_names for k in ["total_time", "new_token"]):
        total_times = [float(t) for t in data["total_time"] if t is not None]
        new_tokens = [float(t) for t in data["new_token"] if t is not None]
        throughputs = [float(t) for t in data["throughput"] if t is not None]

        if len(total_times) > 0 and len(new_tokens) > 0:
            macro_tp = sum(throughputs) / len(throughputs)
            micro_tp = sum(new_tokens) / sum(total_times)
            print(f"Macro average throughput: {macro_tp:.4f} tokens/s")
            print(f"Micro average throughput: {micro_tp:.4f} tokens/s")
        else:
            print("Throughput fields found but empty.")
    else:
        print("No throughput fields detected in file.")

    # ---- 2. Accept length  ----
    if "avg_acc_length" in data.column_names:
        values = [float(v) for v in data["avg_acc_length"] if v is not None]
        if len(values) > 0:
            avg_acc = np.mean(values)
            print(f"Average accept length: {avg_acc:.4f}")
        else:
            print("avg_acc_length field present but empty.")
    else:
        print("No avg_acc_length field detected in file.")

    print("--------------------------------------------------------\n")

if __name__ == "__main__":
    file_paths = [
        "/home/lab6033/zhendong/SpecPV/outputs/llama3.1-8b-instruct/qmsum-specpv-2048-20.jsonl",
        "/home/lab6033/zhendong/SpecPV/outputs/llama3.1-8b-instruct/gov_report-specpv-2048-20.jsonl",
    ]
    for path in file_paths:
        process_outputs(path)
