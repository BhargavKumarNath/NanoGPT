import json
import regex as re
from collections import defaultdict

# Regex for splitting text into words and pinctuations based on GPT-4's patterns
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_pair_counts(ids):
    """Counts the frequency of consecutive paits of token IDs"""
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge_pairs(ids, pair, new_token_id):
    """Replaces all occurences of a pair in a list of IDs with a new token ID"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(new_token_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class BpeTokenizer:
    """A from scratch implementation of a Byte Pair Encoding (BPE) Tokenizer"""

    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.pattern = re.compile(GPT4_SPLIT_PATTERN)
    
    def train(self, corpus_path: str, vocab_size: int, special_tokens: list = None):
        """Train the tokenizer on a given corpus with progress logging"""
        assert vocab_size >= 256
        if special_tokens is None:
            special_tokens = []

        print(f"\n[BPE] Starting training...")
        print(f"[BPE] Corpus: {corpus_path}")
        print(f"[BPE] Target vocab size: {vocab_size}")

        # 1. Initialize vocabulary with bytes and special tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, token in enumerate(special_tokens):
            self.special_tokens[token] = vocab_size + i
            self.vocab[vocab_size + i] = token.encode("utf-8")

        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        # 2. Read Corpus
        print("[BPE] Reading corpus...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"[BPE] Corpus loaded ({len(text)/1e6:.2f}M chars)")

        print("[BPE] Splitting into chunks...")
        chunks = re.findall(self.pattern, text)
        print(f"[BPE] Total chunks: {len(chunks):,}")

        print("[BPE] Encoding chunks to bytes...")
        tokenized_chunks = [list(chunk.encode('utf-8')) for chunk in chunks]
        print(f"[BPE] Byte tokenized.")

        # 3. Iterative merges
        num_merges = vocab_size - 256
        self.merges = {}
        print(f"[BPE] Total merges to perform: {num_merges}")

        for i in range(num_merges):

            if i % 10 == 0:
                print(f"[BPE] Merge step {i}/{num_merges} (scanning pairs...)")

            # Count pairs
            pair_counts = defaultdict(int)
            for chunk_ids in tokenized_chunks:
                # FAST path: skip tiny chunks
                if len(chunk_ids) < 2:
                    continue
                for pair, count in get_pair_counts(chunk_ids).items():
                    pair_counts[pair] += count

            if not pair_counts:
                print("[BPE] No more pairs. Stopping early.")
                break

            # Choose most frequent pair
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            new_token_id = 256 + i

            if i % 10 == 0:
                print(f"[BPE]  • Most frequent pair so far: {most_frequent_pair} → new token {new_token_id}")

            # Apply merge across all chunks
            tokenized_chunks = [
                merge_pairs(chunk_ids, most_frequent_pair, new_token_id)
                for chunk_ids in tokenized_chunks
            ]

            # Record merge
            self.merges[most_frequent_pair] = new_token_id
            self.vocab[new_token_id] = (
                self.vocab[most_frequent_pair[0]] + self.vocab[most_frequent_pair[1]]
            )

            if (i + 1) % 100 == 0:
                print(f"[BPE] ---- {i+1}/{num_merges} merges completed ----")

        # Add special tokens to vocab
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

        print("\n[BPE] Training complete.")


    def _encode_chunk(self, text_bytes: list) -> list:
        """Encodes a single chunk of text that has already been byte-encoded."""
        ids = list(text_bytes)
        while len(ids) > 1:
            pair_counts = get_pair_counts(ids)
            # Find the merge with the lowest rank (earliest learned)
            pair = min(pair_counts, key=lambda p: self.merges.get(p, float('inf')))
            
            if pair not in self.merges:
                break # No more merges found
            
            new_token_id = self.merges[pair]
            ids = merge_pairs(ids, pair, new_token_id)
        return ids

    def encode(self, text: str) -> list:
        """Encodes a string into a list of token IDs."""
        encoded_ids = []
        # Handle special tokens first by splitting the text
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)

        for chunk in special_chunks:
            if chunk in self.special_tokens:
                encoded_ids.append(self.special_tokens[chunk])
            else:
                # Regular text processing
                for word in re.findall(self.pattern, chunk):
                    encoded_ids.extend(self._encode_chunk(word.encode('utf-8')))
        return encoded_ids
    
    def decode(self, ids: list) -> str:
        """Decodes a list of token IDs back into a string."""
        decoded_bytes = b""
        for token_id in ids:
            if token_id in self.vocab:
                 decoded_bytes += self.vocab[token_id]
            elif token_id in self.inverse_special_tokens:
                 decoded_bytes += self.inverse_special_tokens[token_id].encode("utf-8")
            else:
                 raise ValueError(f"Invalid token ID: {token_id}")
        
        return decoded_bytes.decode('utf-8', errors='replace')
    
    def save(self, file_prefix: str):
        """Saves vocab and merges to files."""
        # Save vocab
        # Invert vocab for JSON serialization (bytes cannot be keys)
        inverse_vocab = {v.decode('utf-8', errors='replace'): k for k, v in self.vocab.items()}
        with open(f"{file_prefix}.vocab.json", "w", encoding='utf-8') as f:
            json.dump(inverse_vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges
        with open(f"{file_prefix}.merges.txt", "w", encoding='utf-8') as f:
            for pair, idx in self.merges.items():
                f.write(f"{pair[0]} {pair[1]}\n")
        print(f"Tokenizer saved to {file_prefix}.vocab.json and {file_prefix}.merges.txt")

    def load(self, file_prefix: str):
        """Loads vocab and merges from files."""
        # Load vocab
        with open(f"{file_prefix}.vocab.json", "r", encoding='utf-8') as f:
            inverse_vocab = json.load(f)
            self.vocab = {v: k.encode('utf-8') for k, v in inverse_vocab.items()}

        # Load merges
        self.merges = {}
        with open(f"{file_prefix}.merges.txt", "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                p1, p2 = map(int, line.strip().split())
                self.merges[(p1, p2)] = 256 + i # Reconstruct the merge index
        
        # Re-populate special tokens based on loaded vocab
        # A bit of a hack: assume special tokens are those with non-byte values
        for idx, token_bytes in self.vocab.items():
            if not (len(token_bytes) == 1 and 0 <= idx < 256):
                token_str = token_bytes.decode('utf-8')
                if re.match(r'<\|.*?\|>', token_str):
                    self.special_tokens[token_str] = idx

        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        print(f"Tokenizer loaded from {file_prefix}.vocab.json and {file_prefix}.merges.txt")

