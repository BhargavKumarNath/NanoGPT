import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nano_gpt.model.model import NanoGptModel
from nano_gpt.tokenizer.bpe import BpeTokenizer
from nano_gpt.config.model_config import NanoGptConfig
from torch.serialization import add_safe_globals
add_safe_globals([NanoGptConfig])

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = "out/mha_baseline/final_model.pt"
TOKENIZER_PATH = "out/mha_baseline/tinystories_tokenizer"

# Load tokenizer
tokenizer = BpeTokenizer()
tokenizer.load(TOKENIZER_PATH)

# Load model
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
config = checkpoint['model_config']
model = NanoGptModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("Model loaded successfully")
print(f"Vocab size: {config.vocab_size}")

# Test with a simple prompt
prompt = "Once upon a time"
start_tokens = tokenizer.encode(prompt)
print(f"\nPrompt: '{prompt}'")
print(f"Encoded tokens: {start_tokens}")

start_tokens_tensor = torch.tensor([start_tokens], dtype=torch.long, device=device)

# Generate with detailed logging
print("\n=== Starting Generation with Debugging ===\n")

model.eval()
kv_cache = None
idx = start_tokens_tensor

with torch.no_grad():
    for i in range(20):  # Generate 20 tokens
        idx_cond = idx if i == 0 else idx[:, -1:]
        start_pos = 0 if i == 0 else idx.size(1) - 1
        
        print(f"\n--- Step {i} ---")
        print(f"Input shape: {idx_cond.shape}, start_pos: {start_pos}")
        print(f"Current sequence length: {idx.size(1)}")
        
        # Forward pass
        logits, _, kv_cache = model(
            idx_cond,
            use_cache=True,
            past_kv_cache=kv_cache,
            start_pos=start_pos
        )
        
        # Get logits for last position
        logits = logits[:, -1, :] / 0.8  # temperature
        
        # Check logits distribution
        print(f"Logits shape: {logits.shape}")
        print(f"Logits min/max: {logits.min().item():.2f} / {logits.max().item():.2f}")
        
        # Apply top-k
        top_k = 5
        v, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        print(f"Top-{top_k} values: {v[0].tolist()}")
        print(f"Top-{top_k} indices: {indices[0].tolist()}")
        print(f"Top-{top_k} tokens: {[tokenizer.decode([idx.item()]) for idx in indices[0]]}")
        
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Check probability distribution
        print(f"Probs sum: {probs.sum().item():.6f}")
        top_probs, top_indices = torch.topk(probs, 5)
        print(f"Top-5 probs: {top_probs[0].tolist()}")
        
        # Sample
        idx_next = torch.multinomial(probs, num_samples=1)
        next_token = idx_next[0, 0].item()
        next_token_str = tokenizer.decode([next_token])
        
        print(f"Sampled token: {next_token} -> '{next_token_str}'")
        
        # Append
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Decode current sequence
        current_text = tokenizer.decode(idx[0].tolist())
        print(f"Current text: '{current_text}'")

print("\n=== Final Output ===")
final_text = tokenizer.decode(idx[0].tolist())
print(final_text)