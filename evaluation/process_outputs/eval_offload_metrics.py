import os
import re
import numpy as np
from datasets import load_dataset

file_dir = "outputs/llama3.1-8b-instruct/pg19_speedup"

# match file_name: pg19-10k-naive.jsonl / pg19-10k-specpv-4096-20.jsonl etc.
pattern = re.compile(r"pg19-(\d+k)-(full|specpv-\d+-20)\.jsonl")

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

print("\n=== Micro Average Throughput (tokens/s) ===\n")

header = f"{'Length':<8}{'Full':>10}"
for length in [2048, 4096, 8192]:
    header += f"{f'SpecPV-{length}':>18}"
print(header)
print("-" * len(header))

for group in length_groups:
    full_tp = results.get((group, "full"), {}).get("throughput", float("nan"))
    line = f"{group:<8}{full_tp:>10.2f}"

    for length in [2048, 4096, 8192]:
        key = (group, f"specpv-{length}-20")
        if key in results:
            specpv_tp = results[key]["throughput"]
            line += f"{specpv_tp:>18.2f}"
        else:
            line += f"{'N/A':>18}"
    print(line)

print("\n(Note: Throughput = micro-average tokens/s)\n")
