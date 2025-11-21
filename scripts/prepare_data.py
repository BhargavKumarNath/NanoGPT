import os
import torch
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import multiprocessing

# Config
DATASET_NAME = "HuggingFaceTB/cosmopedia"
SUBSET = "stories"
SUBSET = "stanford"
output_dir = "data_corpus"
os.makedirs(output_dir, exist_ok=True)

def process_shared(args):
    dataset_shard, shard_id = args
    enc = tiktoken.get_encoding("gpt2")

    token_list = []
    for item in tqdm(dataset_shard, position=shard_id, desc=f"Shared {shard_id}"):
        text = item['text']
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        token_list.extend(ids)

    return token_list

def prepare_cosmopedia():
    print(f"Downloading {DATASET_NAME} ({SUBSET})...")
    dataset = load_dataset(DATASET_NAME, SUBSET, split="train")

    # Split for validation
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_data = split_dataset['train']
    val_data = split_dataset['test']

    print(f"Processing {len(train_data)} training docs and {len(val_data)} val docs...")

    # Define encoding
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    # Helper to write binary files
    def write_bin(data_split, filename):
        doc_ids = []
        print(f"Tokenizing {filename}...")
        for doc in tqdm(data_split):
            ids = enc.encode_ordinary(doc['text'])
            ids.append(enc.eot_token)
            doc_ids.extend(ids)
        print(f"Writing {len(doc_ids)} tokens to {filename}...")
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(os.path.join(output_dir, filename), 'wb') as f:
            f.write(arr.tobytes())

    write_bin(train_data, "train.bin")
    write_bin(val_data, "val.bin")

    import json
    meta = {
        'vocab_size': vocab_size,
        'encoding': 'gpt2'
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

if __name__ == "__main__":
    prepare_cosmopedia()

