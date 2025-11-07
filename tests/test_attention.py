import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.model.attention import MultiHeadAttention, GroupedQueryAttention, SlidingWindowAttention

# ---------------------------------
# ----- MultiHeadAttention -----
# ---------------------------------

def test_mha_output_shape():
    """Tests if the MHA module produces the correct output shape."""
    print("--- Testing MHA Output Shape ---")
    
    # --- Configuration ---
    batch_size = 4
    seq_len = 64
    embed_dim = 128
    num_heads = 8

    # --- Setup ---
    # Create a random input tensor
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    
    # Instantiate the MHA module
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # --- Action ---
    output_tensor = mha(input_tensor)
    
    # --- Verification ---
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_tensor.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {output_tensor.shape}"
    print("Status: OK")

def test_mha_with_causal_mask():
    """Tests if the MHA module works correctly with a causal mask."""
    print("\n--- Testing MHA with Causal Mask ---")

    # --- Configuration ---
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    num_heads = 4

    # --- Setup ---
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Create a causal mask. The `triu` function creates an upper triangular matrix.
    # A value of `True` means the position will be masked (set to -inf).
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    # --- Action ---
    output_tensor = mha(input_tensor, mask=causal_mask)

    # --- Verification ---
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Causal mask shape: {causal_mask.shape}")
    print("Causal mask:\n", causal_mask)
    print(f"Output shape: {output_tensor.shape}")

    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_tensor.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {output_tensor.shape}"
    print("Status: OK (shapes are correct)")


# ---------------------------------
# ----- GroupedQueryAttention -----
# ---------------------------------

def test_gqa_output_shape():
    """Tests if the GQA module produces the correct output shape."""
    print("\n--- Testing GQA Output Shape ---")
    
    # --- Configuration ---
    batch_size = 4
    seq_len = 64
    embed_dim = 128
    num_heads = 8
    num_kv_heads = 2 

    # --- Setup ---
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    gqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)
    
    # --- Action ---
    output_tensor = gqa(input_tensor)
    
    # --- Verification ---
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_tensor.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {output_tensor.shape}"
    print("Status: OK")


def test_mqa_as_gqa_special_case():
    """Tests Multi-Query Attention (MQA) as a special case of GQA."""
    print("\n--- Testing MQA (as GQA special case) Output Shape ---")
    
    # --- Configuration ---
    batch_size = 4
    seq_len = 64
    embed_dim = 128
    num_heads = 8
    num_kv_heads = 1 # MQA

    # --- Setup ---
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    mqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)
    
    # --- Action ---
    output_tensor = mqa(input_tensor)
    
    # --- Verification ---
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_tensor.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {output_tensor.shape}"
    print("Status: OK")


# ---------------------------------
# ----- SlidingWindowAttention -----
# ---------------------------------

def test_sliding_window_attention():
    """Tests the SlidingWindowAttention module and inspects its mask."""
    print("\n--- Testing Sliding Window Attention ---")
    
    # --- Configuration ---
    batch_size = 1
    seq_len = 8
    embed_dim = 16
    num_heads = 4
    num_kv_heads = 2
    window_size = 3

    # --- Setup ---
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    swa = SlidingWindowAttention(embed_dim, num_heads, num_kv_heads, window_size)
    
    # --- Action ---
    output_tensor = swa(input_tensor)
    
    # --- Verification ---
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_tensor.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {output_tensor.shape}"
    print("Status: OK (shapes are correct)")

    # --- Mask Inspection ---
    print(f"\nInspecting mask for seq_len={seq_len} and window_size={window_size}")
    # The mask is cached inside the module after the forward pass
    # It has extra dimensions for batch and heads, so squeeze it.
    generated_mask = swa.mask.squeeze()
    print("Generated Mask (True means the position is MASKED):")
    print(generated_mask)

    # Manual check for a few positions
    # Token 4 should attend to tokens 1, 2, 3, 4 (i.e., mask is False at these positions)
    # window is 3, so it attends to (4-3)=1, 2, 3. And itself, 4.
    assert not generated_mask[4, 1] # attends to 1
    assert not generated_mask[4, 2] # attends to 2
    assert not generated_mask[4, 3] # attends to 3
    assert not generated_mask[4, 4] # attends to itself
    assert generated_mask[4, 0], "Token 4 should NOT attend to token 0"
    assert generated_mask[4, 5], "Token 4 should NOT attend to token 5"
    print("\nMask logic verified for sample positions.")
    print("Status: OK")

if __name__ == "__main__":
    test_mha_output_shape()
    test_mha_with_causal_mask()
    test_gqa_output_shape()
    test_mqa_as_gqa_special_case()
    test_sliding_window_attention()
