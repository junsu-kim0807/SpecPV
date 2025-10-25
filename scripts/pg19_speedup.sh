export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="/data/zhendong/hf_home"
export HF_ENDPOINT=https://hf-mirror.com

python evaluation/pg19_speedup.py \
    --model_path "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K" \
    --method "specpv" \
    --partial_length "2048"

python evaluation/pg19_speedup.py \
    --model_path "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K" \
    --method "specpv" \
    --partial_length "4096"

python evaluation/pg19_speedup.py \
    --model_path "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K" \
    --method "specpv" \
    --partial_length "8192"

python evaluation/pg19_speedup.py \
    --model_path "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K" \
    --method "full"

python evaluation/pg19_speedup.py \
    --model_path "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K" \
    --method "naive"