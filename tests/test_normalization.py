import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.model.normalization import RMSNorm

def test_rmsnorm_functionality():
    """Tests if RMSNorm produces the correct shape and has the expected properties"""
    print("Testing RMSNorm")

    # Config
    batch_size = 4
    seq_len = 64
    embed_dim = 128

    # Setup
    input_tensor = torch.randn(batch_size, seq_len, embed_dim) * 10 # scale up to make norm non trivial
    norm_layer = RMSNorm(embed_dim=embed_dim)
    output_tensor = norm_layer(input_tensor)

    # Verification
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_tensor.shape == expected_shape, "Output shape is incorrect"

    # A key property of RMSNorm is that the L2 norm of the normalized vector is sqrt(d). Since we have a learnable gain initialized to 1, the output norm should be close to this value.
    # We calculate the L2 norm for one vector in the batch.
    sample_output_vector = output_tensor[0, 0, :]
    l2_norm = torch.linalg.norm(sample_output_vector)

    expected_norm = math.sqrt(embed_dim)
    print(f"L2 norm of a sample output vector: {l2_norm.item():.4f}")
    print(f"Expected L2 norm (sqrt(embed_dim)): {expected_norm:.4f}")

    assert torch.allclose(l2_norm, torch.tensor(expected_norm), atol=1e-5), \
        "The L2 norm of the output is not as expected."

    print("Status: OK")

if __name__ == "__main__":
    import math
    test_rmsnorm_functionality()




