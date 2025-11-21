import os
import torch
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import json

# Config
DATASET_NAME = "HuggingFaceTB/cosmopedia"
# Pick ONE. 'stories' is huge, 'stanford' is smaller.
SUBSET = "stanford"
output_dir = "data_corpus"
os.makedirs(output_dir, exist_ok=True)

def prepare_cosmopedia():
    print(f"Downloading {DATASET_NAME} ({SUBSET})...")
    # Load in streaming mode or just load normally (cosmopedia stanford fits in memory, stories might not)
    dataset = load_dataset(DATASET_NAME, SUBSET, split="train")

    # Split for validation (1%)
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_data = split_dataset['train']
    val_data = split_dataset['test']

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    def process_and_save(data_split, filename):
        print(f"Processing {filename}...")
        
        # 1. Pre-calculate total token count to allocate disk space
        # This avoids loading a massive list into 16GB RAM
        total_tokens = 0
        print("Counting tokens...")
        for doc in tqdm(data_split, desc="Counting"):
            # Fast approximate count to check size or exact count
            # We do exact here to be safe with memmap
            ids = enc.encode_ordinary(doc['text'])
            total_tokens += len(ids) + 1 # +1 for EOT

        print(f"Total tokens: {total_tokens / 1e6:.2f}M")
        
        # 2. Create a memory-mapped file on disk (Using virtual memory, not RAM)
        path = os.path.join(output_dir, filename)
        arr = np.memmap(path, dtype=np.uint16, mode='w+', shape=(total_tokens,))

        # 3. Write data into the memmap
        idx = 0
        print("Writing to disk...")
        for doc in tqdm(data_split, desc="Writing"):
            ids = enc.encode_ordinary(doc['text'])
            ids.append(enc.eot_token)
            
            # Write directly to disk buffer
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
            
        arr.flush() # Ensure data is written
        print(f"Saved {path}")

    process_and_save(train_data, "train.bin")
    process_and_save(val_data, "val.bin")

    meta = {
        'vocab_size': vocab_size,
        'encoding': 'gpt2',
        'subset': SUBSET
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

if __name__ == "__main__":
    prepare_cosmopedia()