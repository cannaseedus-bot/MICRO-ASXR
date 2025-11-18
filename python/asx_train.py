#!/usr/bin/env python3
"""
ASX Training Script
Supports CPU/GPU training with Colab integration
"""

import json
import sys
import os
from pathlib import Path
import argparse

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    from tqdm import tqdm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch transformers")


class ASXDataset(Dataset):
    """Dataset for ASX language training"""

    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def train_ngram_model(dataset_path, output_dir='brain'):
    """Train simple n-gram model (no PyTorch required)"""

    print(f"Training n-gram model from {dataset_path}")

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    bigrams = {}
    trigrams = {}

    for text in tqdm(texts, desc="Processing"):
        tokens = text.lower().strip().split()

        # Build bigrams
        for i in range(len(tokens) - 1):
            key = tokens[i]
            next_token = tokens[i + 1]

            if key not in bigrams:
                bigrams[key] = {}

            if next_token not in bigrams[key]:
                bigrams[key][next_token] = 0

            bigrams[key][next_token] += 1

        # Build trigrams
        for i in range(len(tokens) - 2):
            key = f"{tokens[i]} {tokens[i + 1]}"
            next_token = tokens[i + 2]

            if key not in trigrams:
                trigrams[key] = {}

            if next_token not in trigrams[key]:
                trigrams[key][next_token] = 0

            trigrams[key][next_token] += 1

    # Save models
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'bigrams.json'), 'w') as f:
        json.dump(bigrams, f, indent=2)

    with open(os.path.join(output_dir, 'trigrams.json'), 'w') as f:
        json.dump(trigrams, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Bigrams: {len(bigrams)} keys")
    print(f"  Trigrams: {len(trigrams)} keys")
    print(f"  Saved to: {output_dir}/")

    return {
        'bigram_keys': len(bigrams),
        'trigram_keys': len(trigrams),
        'output_dir': output_dir
    }


def train_transformer_model(
    dataset_path,
    output_dir='models/asx-gpt2',
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    use_gpu=True
):
    """Train transformer model (requires PyTorch)"""

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install torch transformers")

    print(f"Training transformer model from {dataset_path}")

    # Setup device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(texts)} training examples")

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4
    )

    model = GPT2LMHeadModel(config)
    model.to(device)

    # Create dataset and dataloader
    dataset = ASXDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    total_loss = 0

    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        total_loss += avg_loss

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Training complete!")
    print(f"  Model saved to: {output_dir}/")
    print(f"  Average loss: {total_loss / epochs:.4f}")

    return {
        'output_dir': output_dir,
        'epochs': epochs,
        'final_loss': total_loss / epochs
    }


def main():
    parser = argparse.ArgumentParser(description='Train ASX language model')
    parser.add_argument('dataset', type=str, help='Path to training dataset')
    parser.add_argument('--type', type=str, default='ngram', choices=['ngram', 'transformer'],
                      help='Model type to train')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs (transformer only)')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size (transformer only)')
    parser.add_argument('--lr', type=float, default=5e-5,
                      help='Learning rate (transformer only)')
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU if available')

    args = parser.parse_args()

    # Check dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)

    # Set default output directory
    if args.output is None:
        args.output = 'brain' if args.type == 'ngram' else 'models/asx-gpt2'

    # Train model
    try:
        if args.type == 'ngram':
            result = train_ngram_model(args.dataset, args.output)
        else:
            result = train_transformer_model(
                args.dataset,
                args.output,
                args.epochs,
                args.batch_size,
                args.lr,
                args.gpu
            )

        # Output result as JSON for API consumption
        print("\nResult:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()