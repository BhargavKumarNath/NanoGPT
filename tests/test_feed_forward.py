import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.model.feed_forward import SwiGLUFeedForward

def test_swiglu_shape_and_functionality():
    """Tests if the SwiGLU FFN returns the correct shape"""
    print(f"Testing SwiGLU Feed-Forward Network")

    # Config
    batch_size = 4
    seq_len = 64
    embed_dim = 128

    # Setup
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    ffn = SwiGLUFeedForward(embed_dim=embed_dim, dropout=0.1)

    output_tensor = ffn(input_tensor)

    # Verification
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_tensor.shape == expected_shape, "Output shape is incorrect"

    # Check to ensure output is not just 0's or the input
    assert not torch.allclose(input_tensor, output_tensor), "Output should be different from input"
    assert not torch.allclose(torch.zeros_like(output_tensor), output_tensor), "Output should not be all zeros"

    print("Status: OK")

if __name__ == "__main__":
    test_swiglu_shape_and_functionality()