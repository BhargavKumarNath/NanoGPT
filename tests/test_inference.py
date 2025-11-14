import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.config.model_config import NanoGptConfig
from nano_gpt.model.model import NanoGptModel
from nano_gpt.tokenizer.bpe import BpeTokenizer

def test_generation_with_kv_cache():
    """Tests the model's generate method"""
    print("Testing Text Generation with KV Cache")

    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = NanoGptConfig(
        vocab_size=301,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        seq_len=32,
        attention_type='mha'
    )

    model = NanoGptModel(config).to(device)
    
    # Create a dummy starting context
    start_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device) # Batch size of 1
    
    # Action
    print(f"Starting context: {start_tokens.tolist()}")
    max_new_tokens = 10
    generated_tokens = model.generate(start_tokens, max_new_tokens=max_new_tokens, temperature=0.8, top_k=5)
    
    # Verification
    print(f"Generated sequence: {generated_tokens.tolist()}")
    
    expected_length = start_tokens.shape[1] + max_new_tokens
    assert generated_tokens.shape[1] == expected_length, "Generated sequence has incorrect length"
    
    # A simple check to ensure it's not just repeating the input
    assert not torch.equal(generated_tokens[:, :start_tokens.shape[1]], generated_tokens[:, -start_tokens.shape[1]:]), \
        "Generated output should not be a simple repetition of the input."

    print("\nGeneration finished with expected length.")
    print("Status: OK")

if __name__ == "__main__":
    test_generation_with_kv_cache()