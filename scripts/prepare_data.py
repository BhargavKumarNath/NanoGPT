import os
from datasets import load_dataset

def prepare_tinystories():
    os.makedirs("data_corpus", exist_ok=True)

    train_file = "data_corpus/tinystories.txt"
    val_file   = "data_corpus/tinystories_val.txt"

    # TRAIN SET (100k stories)
    if not os.path.exists(train_file):
        print("📥 Downloading TinyStories train split...")
        train_dataset = load_dataset("roneneldan/TinyStories", split="train")

        print("✍️ Writing 100k training stories...")
        with open(train_file, "w", encoding="utf-8") as f:
            num_stories = min(100000, len(train_dataset))
            for i in range(num_stories):
                if i % 500 == 0:
                    print(f" ... {i}/{num_stories}")
                f.write(train_dataset[i]["text"])
                f.write("\n<|endoftext|>\n")
        print("✅ Training file ready!")
    else:
        print(f"⚠️ '{train_file}' already exists. Skipping.")

    # VALIDATION SET (1k stories)
    if not os.path.exists(val_file):
        print("📥 Downloading TinyStories validation split...")
        val_dataset = load_dataset("roneneldan/TinyStories", split="validation")

        print("✍️ Writing 1k validation stories...")
        with open(val_file, "w", encoding="utf-8") as f:
            num_val = min(1000, len(val_dataset))
            for i in range(num_val):
                f.write(val_dataset[i]["text"])
                f.write("\n<|endoftext|>\n")
        print("✅ Validation file ready!")
    else:
        print(f"⚠️ '{val_file}' already exists. Skipping.")


if __name__ == "__main__":
    prepare_tinystories()
