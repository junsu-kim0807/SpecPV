import pandas as pd
import pyarrow.parquet as pq
import random
import re
from transformers import AutoTokenizer
import math
from tqdm import tqdm

def load_parquet_files(file_paths):
    dfs = []
    for path in file_paths:
        try:
            table = pq.read_table(path)
            df = table.to_pandas()
            dfs.append(df)
            print(f"Successfully loaded {path}, containing {len(df)} entries")
        except Exception as e:
            print(f"Failed to load {path}: {str(e)}")
    return pd.concat(dfs, ignore_index=True)


def find_last_sentence_end(text):
    """Find the position of the last sentence end in the text (based on English punctuation)"""
    # Match sentence-ending punctuation: . ! ? possibly followed by space and uppercase letter
    pattern = r'[.!?]\s*[A-Z]'
    matches = list(re.finditer(pattern, text))
    
    if matches:
        # Return the position of the last match
        last_match = matches[-1]
        return last_match.start() + 1  # Add 1 to include the punctuation itself
    else:
        # If no sentence-ending markers found, return the length of the text
        return len(text)


def filter_and_truncate_text(df, text_column, target_lengths, tokenizer, samples_per_length=20):
    """
    Filter and truncate text data, ensuring each original text is used only once (based on token count)
    and truncated to complete sentences
    """
    # Calculate token count for each text, add progress bar
    print("Calculating text token counts...")
    tqdm.pandas()
    df['token_length'] = df[text_column].progress_apply(
        lambda x: len(tokenizer.encode(x, truncation=False, add_special_tokens=False))
    )

    # Filter texts with sufficient token counts (slightly increase requirement to ensure complete sentences)
    eligible_data = df[df['token_length'] >= 10240][[text_column, 'token_length']]
    eligible_data = eligible_data.sort_values(by='token_length', ascending=True)
    eligible_texts = eligible_data[text_column].tolist()
    total_needed = samples_per_length * len(target_lengths)
    
    # Check if there are enough eligible texts
    if len(eligible_texts) < total_needed:
        raise ValueError(
            f"Insufficient number of eligible texts! Need {total_needed} entries, only found {len(eligible_texts)} entries"
        )
    

    result = []
    for i, length in enumerate(target_lengths):

        # Ensure indices are within valid range
        start_idx = i * samples_per_length
        end_idx = start_idx + samples_per_length
        selected = eligible_texts[start_idx:end_idx]
        
        # Truncate texts to target token length and add to results
        print(f"Processing target length: {length} tokens")
        for text in tqdm(selected, desc=f"Processing {length} tokens samples"):
            # Encode to tokens without adding special tokens
            tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
            
            # First truncate to slightly more than target length to find complete sentences
            initial_cut = min(length + 200, len(tokens)) 
            truncated_tokens = tokens[:initial_cut]
            
            # Decode back to text
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            # Find position of last sentence end
            sentence_end_pos = find_last_sentence_end(truncated_text)
            
            final_text = truncated_text[:sentence_end_pos]            
            result.append({
                'length': length,
                'text': final_text
            })
    
    return pd.DataFrame(result)

if __name__ == "__main__":
    FILE_PATHS = [
        "/home/lab6033/hcy/SpecPV/data/pg-19/test-00000-of-00001-29a571947c0b5ccc.parquet",
        "/home/lab6033/hcy/SpecPV/data/pg-19/train-00022-of-00023-5a956eb2a5d6cab5.parquet"
    ]
    TEXT_COLUMN = "text"  
    TARGET_LENGTHS = [10240, 20480, 30720, 40960, 51200, 61440]
    SAMPLES_PER_LENGTH = 100
    OUTPUT_FILE = "/home/lab6033/hcy/SpecPV/data/pg-19/test_pg19_600_sentences.parquet"
    MODEL_NAME = '/home/lab6033/zhendong/models/LLAMA3.1-8B-Instruct'  
    
    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load data
    print("Starting to load data...")
    combined_df = load_parquet_files(FILE_PATHS)
    print(f"Total data entries: {len(combined_df)}")
    
    # Filter and truncate texts
    print("Starting to filter and truncate texts...")
    extracted_df = filter_and_truncate_text(
        combined_df, 
        TEXT_COLUMN, 
        TARGET_LENGTHS, 
        tokenizer,
        SAMPLES_PER_LENGTH
    )
    
    # Save results
    extracted_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Extraction completed, total {len(extracted_df)} entries saved to {OUTPUT_FILE}")
    
    # Print statistics
    print("\nExtraction result statistics:")
    for length in TARGET_LENGTHS:
        subset = extracted_df[extracted_df['length'] == length]
        count = len(subset)
        print(f"Target length {length} tokens: {count} entries")