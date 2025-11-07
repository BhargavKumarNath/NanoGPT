import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.model.positional_encoding import RotaryPositionalEmbeddings
from nano_gpt.model.attention import MultiHeadAttention

def test_rope_shape():
    """Tests if RoPE returns tensors of the correct shape."""
    print("--- Testing RoPE Shape ---")
    
    # --- Config ---
    batch_size = 4
    num_heads = 8
    seq_len = 64
    head_dim = 16

    # --- Setup ---
    rope = RotaryPositionalEmbeddings(dim=head_dim)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # --- Action ---
    q_out, k_out = rope(q, k)

    # --- Verification ---
    print(f"Input Q shape:  {q.shape}")
    print(f"Output Q shape: {q_out.shape}")
    print(f"Input K shape:  {k.shape}")
    print(f"Output K shape: {k_out.shape}")
    
    assert q.shape == q_out.shape, "Query shape mismatch"
    assert k.shape == k_out.shape, "Key shape mismatch"
    print("Status: OK")

def test_rope_integration_with_mha():
    """Tests if MHA works correctly when integrated with RoPE."""
    print("\n--- Testing RoPE Integration with MHA ---")

    # --- Config ---
    batch_size = 4
    seq_len = 64
    embed_dim = 128
    num_heads = 8
    
    # head_dim must be even for RoPE
    head_dim = embed_dim // num_heads
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"

    # --- Setup ---
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    
    # Instantiate RoPE and MHA, then pass RoPE to MHA
    rope = RotaryPositionalEmbeddings(dim=head_dim)
    mha_with_rope = MultiHeadAttention(embed_dim, num_heads, rope=rope)

    # --- Action ---
    output_tensor = mha_with_rope(input_tensor)

    # --- Verification ---
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_tensor.shape == expected_shape, "Output shape is incorrect"
    print("Status: OK")

if __name__ == "__main__":
    test_rope_shape()
    test_rope_integration_with_mha()
