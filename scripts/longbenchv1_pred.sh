export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="/data/zhendong/hf_home"
export HF_ENDPOINT=https://hf-mirror.com


python evaluation/longbenchv1_pred.py \
    --dataset "musique" \
    --base_model "/home/lthpc/nvmessd/zhendong/models/Qwen3-8B" \
    --EAGLE_model "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-Qwen3-8B-YARN-64K" \
    --method "full" 

python evaluation/longbenchv1_pred.py \
    --dataset "musique" \
    --base_model "/home/lthpc/nvmessd/zhendong/models/Qwen3-8B" \
    --EAGLE_model "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-Qwen3-8B-YARN-64K" \
    --method "specpv" \
    --partial_length "512"

python evaluation/longbenchv1_pred.py \
    --dataset "musique" \
    --base_model "/home/lthpc/nvmessd/zhendong/models/Qwen3-8B" \
    --EAGLE_model "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-Qwen3-8B-YARN-64K" \
    --method "specpv" \
    --partial_length "1024"

python evaluation/longbenchv1_pred.py \
    --dataset "musique" \
    --base_model "/home/lthpc/nvmessd/zhendong/models/Qwen3-8B" \
    --EAGLE_model "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-Qwen3-8B-YARN-64K" \
    --method "specpv" \
    --partial_length "2048"

python evaluation/longbenchv1_pred.py \
    --dataset "musique" \
    --base_model "/home/lthpc/nvmessd/zhendong/models/Qwen3-8B" \
    --EAGLE_model "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-Qwen3-8B-YARN-64K" \
    --method "specpv" \
    --partial_length "4096"

python evaluation/longbenchv1_pred.py \
    --dataset "musique" \
    --base_model "/home/lthpc/nvmessd/zhendong/models/Qwen3-8B" \
    --EAGLE_model "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-Qwen3-8B-YARN-64K" \
    --method "specpv" \
    --partial_length "8192"








