import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_gpt.tokenizer.bpe import BpeTokenizer

def main():
    # 1. Setup
    corpus_path = "data_corpus/sample.txt"
    vocab_size = 300 
    special_tokens = ["<|endoftext|>"]
    model_prefix = "bpe_tokenizer" 

    # Ensure the sample corpus exists
    if not os.path.exists(corpus_path):
        print(f"Corpus file not found at {corpus_path}")
        print("Please create it with some sample text.")
        return

    # 2. Training
    print("--- Training Tokenizer ---")
    tokenizer = BpeTokenizer()
    tokenizer.train(corpus_path, vocab_size, special_tokens=special_tokens)
    
    # 3. Save the trained tokenizer
    tokenizer.save(model_prefix)
    print("\n--- Tokenizer Saved ---")

    # 4. Verification
    print("\n--- Verifying Saved and Loaded Tokenizer ---")
    
    # Load the tokenizer from the saved files
    loaded_tokenizer = BpeTokenizer()
    loaded_tokenizer.load(model_prefix)
    
    test_sentences = [
        "Hello world!",
        "repetition helps learning",
        "This is a test with a special token: <|endoftext|>",
        "Let's test encoding and decoding."
    ]

    for text in test_sentences:
        print(f"\nOriginal: '{text}'")
        
        # Test loaded tokenizer
        encoded = loaded_tokenizer.encode(text)
        decoded = loaded_tokenizer.decode(encoded)
        
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        
        # Assert that the process is reversible
        assert text == decoded, f"Mismatch! '{text}' != '{decoded}'"
        print("Status: OK")

if __name__ == "__main__":
    main()