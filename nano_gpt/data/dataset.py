import torch
from torch.utils.data import IterableDataset

class StreamingTextDataset(IterableDataset):
    """
    An iterable style dataset for streaming large text corpora.
    
    This dataset reads a text file, tokenizes it, and yields sequences of a fixed length.
    """
    def __init__(self, tokenizer, corpus_path: str, seq_len: int):
        self.tokenizer = tokenizer
        self.corpus_path = corpus_path
        self.seq_len = seq_len

    def __iter__(self):
        """
        The iterator method that yields training examples.
        """
        # Open the text file for reading
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            # Read the entire corpus.
            text = f.read()

        # Tokenize the entire text corpus
        all_token_ids = self.tokenizer.encode(text)

        # Convert to a tensor for easy slicing
        all_token_ids = torch.tensor(all_token_ids, dtype=torch.long)

        # Yield sequences of length seq_len + 1
        for i in range(0, len(all_token_ids) - self.seq_len, self.seq_len):
            chunk = all_token_ids[i : i + self.seq_len + 1]
            
            # The input is the first `seq_len` tokens
            x = chunk[:-1]
            
            # The target is the last `seq_len` tokens (input shifted by one)
            y = chunk[1:]
            
            yield x, y