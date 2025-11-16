import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import sys
import math
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.model.model import NanoGptModel
from nano_gpt.tokenizer.bpe import BpeTokenizer
from nano_gpt.data.dataset import StreamingTextDataset
from nano_gpt.config.model_config import NanoGptConfig

from torch.serialization import add_safe_globals
add_safe_globals([NanoGptConfig])  

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
seq_len = 128

def prepare_validation_data():
    """Downloads validation split of TinyStories"""
    os.makedirs("data_corpus", exist_ok=True)
    output_filename = "data_corpus/tinystories_val.txt"
    
    if os.path.exists(output_filename):
        return output_filename

    print("Downloading TinyStories Validation split...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    
    print("Preparing validation file...")
    with open(output_filename, "w", encoding="utf-8") as f:
        for i in range(500):
            f.write(dataset[i]['text'])
            f.write("\n<|endoftext|>\n")
    
    return output_filename

def evaluate_model(checkpoint_path, tokenizer_path, data_path):
    print(f"\n--- Evaluating Model: {checkpoint_path} ---")
    
    # 1. Load Tokenizer
    tokenizer = BpeTokenizer()
    if not os.path.exists(tokenizer_path + ".vocab.json"):
         print(f"Error: Tokenizer not found at {tokenizer_path}")
         return
    tokenizer.load(tokenizer_path)
    
    # 2. Load Model
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['model_config']
    
    model = NanoGptModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: Layers={config.num_layers}, Heads={config.num_heads}, Attn={config.attention_type}")

    # 3. Prepare Data
    dataset = StreamingTextDataset(tokenizer, data_path, config.seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # 4. Evaluation Loop
    total_loss = 0.0
    total_batches = 0
    
    print("Running evaluation loop...")
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass only
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x, y)
            
            total_loss += loss.item()
            total_batches += 1
            
            if total_batches % 50 == 0:
                print(f"Processed {total_batches} batches...")

    if total_batches == 0:
        print("Error: No data processed.")
        return

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    
    print("\n" + "="*40)
    print(f"RESULTS for {checkpoint_path}")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity:      {perplexity:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    val_data_path = prepare_validation_data()
    
    # Evaluate MHA Baseline
    mha_ckpt = "out/mha_baseline/final_model.pt"
    mha_tok = "out/mha_baseline/tinystories_tokenizer"
    evaluate_model(mha_ckpt, mha_tok, val_data_path)
    
    # Evaluate GQA Model
    gqa_ckpt = "out/gqa/final_model.pt"
    gqa_tok = "out/gqa/tinystories_tokenizer"
    evaluate_model(gqa_ckpt, gqa_tok, val_data_path)
