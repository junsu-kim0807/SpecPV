export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="/data/zhendong/hf_home"
export HF_ENDPOINT=https://hf-mirror.com

python evaluation/longbenchv1_summarization.py \
    --model_path "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K" \
    --dataset_name "qmsum" \
    --method "full"

python evaluation/longbenchv1_summarization.py \
    --model_path "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K" \
    --dataset_name "qmsum" \
    --method "naive"

python evaluation/longbenchv1_summarization.py \
    --model_path "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K" \
    --dataset_name "qmsum" \
    --method "specpv" \
    --partial_length "2048" \
    --partial_spec_tokens "20"

python evaluation/longbenchv1_summarization.py \
    --model_path "/home/lthpc/nvmessd/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K" \
    --dataset_name "gov_report" \
    --method "specpv" \
    --partial_length "2048" \
    --partial_spec_tokens "20"
