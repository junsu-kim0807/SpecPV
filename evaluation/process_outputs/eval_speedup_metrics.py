import os
import re
import numpy as np
from datasets import load_dataset

file_dir = "outputs/llama3.1-8b-instruct/pg19_speedup"

# match file_name: pg19-10k-naive.jsonl / pg19-10k-specpv-4096-20.jsonl etc.
pattern = re.compile(r"pg19-(\d+k)-(naive|full|specpv-\d+-20)\.jsonl")

results = {}

for fname in os.listdir(file_dir):
    m = pattern.match(fname)
    if not m:
        continue
    length_group, method = m.groups()
    path = os.path.join(file_dir, fname)

    try:
        ds = load_dataset("json", data_files=path)["train"]
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        continue

    total_times = [float(t) for t in ds["total_time"] if t is not None]
    new_tokens = [float(t) for t in ds["new_token"] if t is not None]
    if len(total_times) == 0 or len(new_tokens) == 0:
        print(f"Skipping {fname}: empty data")
        continue

    micro_tp = sum(new_tokens) / sum(total_times)

    # get avg_acc_length (except for naive method)
    avg_acc = None
    if "avg_acc_length" in ds.column_names:
        values = [float(v) for v in ds["avg_acc_length"] if v is not None]
        if len(values) > 0:
            avg_acc = float(np.mean(values))

    results[(length_group, method)] = {
        "throughput": micro_tp,
        "avg_acc": avg_acc,
    }

# group all lengths
length_groups = sorted({k[0] for k in results.keys()},
                       key=lambda x: int(x.replace("k", "000")))

print("\n=== Micro Average Throughput, Speedup & Avg Accept Length ===\n")
header = f"{'Length':<8}{'Naive(t/s)':>12}{'Full×':>8}{'AccLen':>10}"
for length in [2048, 4096, 8192]:
    header += f"{f'SpecPV-{length}×':>15}{'AccLen':>10}"
print(header)
print("-" * len(header))

for group in length_groups:
    naive = results.get((group, "naive"), {}).get("throughput", float("nan"))
    full_tp = results.get((group, "full"), {}).get("throughput", float("nan"))
    full_acc = results.get((group, "full"), {}).get("avg_acc", None)
    full_acc_str = f"{full_acc:.2f}" if full_acc is not None else "N/A"

    line = f"{group:<8}{naive:>12.2f}{full_tp/naive:>8.2f}{full_acc_str:>10}"

    for length in [2048, 4096, 8192]:
        key = (group, f"specpv-{length}-20")
        if key in results:
            specpv_tp = results[key]["throughput"]
            ratio = specpv_tp / naive
            acc = results[key]["avg_acc"]
            acc_str = f"{acc:.2f}" if acc is not None else "N/A"
            line += f"{ratio:>15.2f}{acc_str:>10}"
        else:
            line += f"{'N/A':>15}{'N/A':>10}"
    print(line)

print("\n(Note: Throughput = micro-average tokens/s; speedups = throughput / naive; AccLen = avg_accept_length)\n")
