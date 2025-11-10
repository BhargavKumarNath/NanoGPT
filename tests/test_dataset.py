# tests/test_dataset.py
import torch
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_gpt.tokenizer.bpe import BpeTokenizer
from nano_gpt.data.dataset import StreamingTextDataset

def test_streaming_dataset_and_dataloader():
    """
    Tests the StreamingTextDataset and its integration with DataLoader.
    """
    print("Testing Streaming Dataset and DataLoader")
    
    # Config
    corpus_path = "data_corpus/sample.txt"
    tokenizer_prefix = "bpe_tokenizer"
    seq_len = 8
    batch_size = 2

    # Setup
    # 1. Load the tokenizer
    assert os.path.exists(f"{tokenizer_prefix}.vocab.json"), "Tokenizer model not found. Run tokenizer test first."
    tokenizer = BpeTokenizer()
    tokenizer.load(tokenizer_prefix)

    # 2. Instantiate the dataset
    dataset = StreamingTextDataset(tokenizer, corpus_path, seq_len)
    
    # 3. Instantiate the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)
    print("Dataset and DataLoader instantiated successfully")

    # Action & Verification
    num_batches = 0
    for i, (x_batch, y_batch) in enumerate(dataloader):
        num_batches += 1
        print(f"\n Batch {i+1} ")
        print(f"x_batch shape: {x_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")

        # Dynamically handle last partial batch
        expected_x_shape = (x_batch.shape[0], seq_len)
        expected_y_shape = (y_batch.shape[0], seq_len)
        assert x_batch.shape == expected_x_shape, "x_batch shape is incorrect"
        assert y_batch.shape == expected_y_shape, "y_batch shape is incorrect"

        # y should be x shifted
        x_sample = x_batch[0]
        y_sample = y_batch[0]
        assert torch.equal(x_sample[1:], y_sample[:-1]), "Target y is not a shifted version of input x"
        print("x/y shift logic is correct")

    assert num_batches > 0, "DataLoader did not yield any batches"
    print(f"\nSuccessfully iterated through {num_batches} batches")
    print("Status: OK")

if __name__ == "__main__":
    test_streaming_dataset_and_dataloader()