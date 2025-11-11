import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_gpt.config.model_config import NanoGptConfig
from nano_gpt.model.model import NanoGptModel

def test_full_model_forward_pass():
    """Tests the forward pass of the complete NanoGptModel"""
    print("Testing Full Model Assembly and Forward Pass")

    # Config
    config = NanoGptConfig(
        vocab_size=300,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        seq_len=32,
        attention_type='mha'
    )

    # Setup
    model = NanoGptModel(config)
    model.eval()
    print("Model instantiated successfully:")
    print(model)

    # Create dummy input data
    batch_size = 4
    input_ids = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))

    # Action
    # Test forward pass with targets 
    logits, loss = model(input_ids, targets)

    # Test forward pass without targets
    logits_only, loss_only = model(input_ids)

    # Verification
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")

    expected_logits_shape = (batch_size, config.seq_len, config.vocab_size)
    assert logits.shape == expected_logits_shape, "Logits shape is incorrect"
    assert loss is not None, "Loss should be calculated when targets are provided"
    assert isinstance(loss.item(), float), "Loss should be a floating point number"

    print(f"Calculated loss: {loss.item():.4f}")

    assert logits_only.shape == expected_logits_shape, "Logits (only) shape is incorrect"
    assert loss_only is None, "Loss should be None when targets are not provided"

    print("\nForward pass with and without targets works as expected")
    print("Status: OK")

if __name__ == "__main__":
    test_full_model_forward_pass()