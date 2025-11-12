import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.model.attention import MultiHeadAttention
from nano_gpt.model.positional_encoding import RotaryPositionalEmbeddings
from nano_gpt.model.feed_forward import SwiGLUFeedForward
from nano_gpt.model.normalization import RMSNorm
from nano_gpt.model.block import TransformerBlock

def test_transformer_block_assembly():
    """
    Tests the assembly and forward pass of a complete TransformerBlock
    """
    print("Testing Transformer Block Assembly")

    # Config
    batch_size = 4
    seq_len = 64
    embed_dim = 128
    num_heads = 8
    head_dim = embed_dim//num_heads

    # Instantiate all sub-components
    # 1. Positional Encoding
    rope = RotaryPositionalEmbeddings(dim=head_dim)

    # 2. Attention Mechanism (with RoPE)
    attention_module = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        rope=rope
    )

    # 3. Feed Forward Network
    ffn_module = SwiGLUFeedForward(embed_dim=embed_dim)

    # 4. Normalization Class
    norm_class = RMSNorm

    # Instantiate the Transformer Block
    transformer_block = TransformerBlock(
        embed_dim=embed_dim,
        attn=attention_module,
        ffn=ffn_module,
        norm_cls=norm_class
    )

    print("Transformer Block instantiated successfully")
    print(transformer_block)

    # Action
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    output_tensor = transformer_block(input_tensor)

    # Verification
    print(f"\nInput Shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_tensor.shape == expected_shape, "Output shape is incorrect"
    assert not torch.allclose(input_tensor, output_tensor), "Output should be different from input"

    print(f"Status: OK")

def test_transformer_block_with_droppath():
    """Tests that DropPath works correctly in both train and eval modes."""
    print("\n--- Testing Transformer Block with DropPath ---")

    # Config
    batch_size, seq_len, embed_dim, num_heads = 2, 8, 16, 4
    head_dim = embed_dim // num_heads
    drop_path_rate = 0.5 

    # Setup
    input_tensor = torch.ones(batch_size, seq_len, embed_dim)
    
    attention_module = MultiHeadAttention(embed_dim, num_heads)
    ffn_module = SwiGLUFeedForward(embed_dim)

    # Instantiate the Block with DropPath
    block = TransformerBlock(embed_dim, attention_module, ffn_module, RMSNorm, drop_path_rate)

    # Test in EVAL mode
    block.eval()
    with torch.no_grad():
        output_eval_1 = block(input_tensor.clone())
        output_eval_2 = block(input_tensor.clone())
    
    print("In eval mode: Two forward passes produce identical results.")
    assert torch.allclose(output_eval_1, output_eval_2), "Outputs should be deterministic in eval mode"

    # Test in TRAIN mode
    block.train()
    output_train_1 = block(input_tensor.clone())
    output_train_2 = block(input_tensor.clone())

    print("In train mode: Two forward passes produce different results (stochastic).")
    assert not torch.allclose(output_train_1, output_train_2), "Outputs should be stochastic in train mode"

    # Check if the output is different from a block with no DropPath
    block_no_drop = TransformerBlock(embed_dim, attention_module, ffn_module, RMSNorm, 0.0)
    block_no_drop.eval() 
    output_no_drop = block_no_drop(input_tensor.clone())

    print("In train mode: Output is different from a block with no DropPath.")
    assert not torch.allclose(output_train_1, output_no_drop), "DropPath should alter the output in train mode"

    print("Status: OK")

if __name__ == "__main__":
    test_transformer_block_assembly()
    test_transformer_block_with_droppath()