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

if __name__ == "__main__":
    test_transformer_block_assembly()