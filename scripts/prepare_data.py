import os
from datasets import load_dataset
import tiktoken

def download_and_prepare():
    """Downloads TinyStories and prepares a training text file"""
    os.makedirs("data_corpus", exist_ok=True)

    output_filename = "data_corpus/tinystories.txt"

    if os.path.exists(output_filename):
        print(f"'{output_filename}' already exists. Skipping download")
        return
    
    print("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    print("Preparing data file...")
    num_stories = 50000

    with open(output_filename, "w", encoding="utf-8") as f:
        for i in range(num_stories):
            if i % 500 == 0:
                print(f" ... processing story {i}/{num_stories}")
            f.write(dataset[i]['text'])
            # Add a seperator to be sure
            f.write("\n<|endoftext|>\n")
    print(f"Successfully created '{output_filename}'")

if __name__ == "__main__":
    download_and_prepare()