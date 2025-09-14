#!/usr/bin/env python3
"""
Prepare datasets for GPT-2 training.
Downloads and processes text datasets into binary format.
"""

import os
import sys
import numpy as np
import requests
from pathlib import Path
import tiktoken

def download_shakespeare():
    """Download and prepare Shakespeare dataset."""
    data_dir = Path("gpt2/data/shakespeare")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download Shakespeare text
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    input_file = data_dir / "input.txt"
    if not input_file.exists():
        print(f"Downloading Shakespeare dataset from {url}...")
        response = requests.get(url)
        with open(input_file, 'w') as f:
            f.write(response.text)
        print(f"Downloaded to {input_file}")
    else:
        print(f"Shakespeare dataset already exists at {input_file}")

    # Read the text
    with open(input_file, 'r') as f:
        text = f.read()

    # Tokenize using GPT-2 tokenizer
    print("Tokenizing dataset...")
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode(text, allowed_special={"<|endoftext|>"})
    print(f"Total tokens: {len(train_ids):,}")

    # Split into train and val (90/10 split)
    split_idx = int(len(train_ids) * 0.9)
    train_data = np.array(train_ids[:split_idx], dtype=np.uint16)
    val_data = np.array(train_ids[split_idx:], dtype=np.uint16)

    # Save as binary files
    train_file = data_dir / "train.bin"
    val_file = data_dir / "val.bin"

    train_data.tofile(train_file)
    val_data.tofile(val_file)

    print(f"Saved {len(train_data):,} training tokens to {train_file}")
    print(f"Saved {len(val_data):,} validation tokens to {val_file}")
    print("Dataset preparation complete!")

def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Prepare datasets for GPT-2 training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        choices=["shakespeare"],
        help="Dataset to prepare"
    )

    args = parser.parse_args()

    if args.dataset == "shakespeare":
        download_shakespeare()
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

if __name__ == "__main__":
    main()