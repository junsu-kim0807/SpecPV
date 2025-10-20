import pandas as pd
import pyarrow.parquet as pq
import json
import re
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def merge_json_files(input_folder, output_file):
    json_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith('.json') and os.path.isfile(os.path.join(input_folder, f))
    ]
    
    all_texts = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    all_texts.append({"text": item["text"]})
           
        except Exception:
            continue

    #print(len(all_texts))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts[:6400], f)


def find_last_sentence_end(text):
    pattern = r'[.!?]\s*[A-Z]'
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].start() + 1
    return len(text)


def process_single_file(df, text_column, target_length, tokenizer):
    buffer = 1500
    result = []

    for text in tqdm(df[text_column], desc="Processing", leave=False):
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
        
        if len(tokens) >= target_length + buffer:
            processed = _process_text(target_length, tokens, tokenizer, buffer)
            result.append(processed)
    
    return pd.DataFrame(result)


def _process_text(target_length, tokens, tokenizer, buffer):
    truncated_tokens = tokens[:target_length + buffer]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    sentence_end_pos = find_last_sentence_end(truncated_text)
    final_text = truncated_text[:sentence_end_pos]
    return {
        'text': final_text
        }


def process_multiple_files(input_files, output_dir, text_column, target_length, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = [
        os.path.join(input_files, filename)
        for filename in os.listdir(input_files)
        if os.path.isfile(os.path.join(input_files, filename)) 
    ]

    for file_path in all_files:
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            result_df = process_single_file(df, text_column, target_length, tokenizer)
            count = len(result_df)
            
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            #parquet_output = os.path.join(output_dir, f"{file_name}_64k.parquet")
            json_output = os.path.join(output_dir, f"{file_name}_64k.json")
            
            #result_df.to_parquet(parquet_output, index=False)
            result_df.to_json(json_output, orient="records", force_ascii=False)
            
            print(f"{file_name}: {count}")
            
        except Exception as e:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"{file_name}: 0")


if __name__ == "__main__":
    INPUT_FILES = "C:/Users/Lenovo/Desktop/pg19/train/new"
    OUTPUT_DIR = "C:/Users/Lenovo/Desktop/pg19/pg19_64k"
    OUTPUT_FILE = "C:/Users/Lenovo/Desktop/pg19/train_pg19_64k_6400.json"
    MODEL_NAME = "D:/model/LLAMA3.1-8B-Instruct"
    TEXT_COLUMN = "text"  
    TARGET_LENGTH = 65536
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    process_multiple_files(
        input_files=INPUT_FILES,
        output_dir=OUTPUT_DIR,
        text_column=TEXT_COLUMN,
        target_length=TARGET_LENGTH,
        tokenizer=tokenizer
    )
    

    merge_json_files(OUTPUT_DIR, OUTPUT_FILE)
    
