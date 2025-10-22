export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="/data/zhendong/hf_home"
export HF_ENDPOINT=https://hf-mirror.com

# python test/test.py
python evaluation/longbenchv1_summarization.py \
    --model_path "/home/lab6033/zhendong/models/LLAMA3.1-8B-Instruct" \
    --draft_path "/home/lab6033/zhendong/SparseSD/data/llama3-8b-eagle3-64k" \
    --dataset_name "gov_report" \
    --method "full"
