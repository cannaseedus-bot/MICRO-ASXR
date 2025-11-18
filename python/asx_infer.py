#!/usr/bin/env python3
"""
ASX Inference Script
Run predictions with trained models
"""

import json
import sys
import os
import argparse

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def predict_ngram(text, brain_dir='brain', max_results=5):
    """Predict next word using n-gram model"""

    # Load brain data
    bigrams_path = os.path.join(brain_dir, 'bigrams.json')
    trigrams_path = os.path.join(brain_dir, 'trigrams.json')

    bigrams = {}
    trigrams = {}

    if os.path.exists(bigrams_path):
        with open(bigrams_path, 'r') as f:
            bigrams = json.load(f)

    if os.path.exists(trigrams_path):
        with open(trigrams_path, 'r') as f:
            trigrams = json.load(f)

    # Tokenize input
    tokens = text.lower().split()
    predictions = []

    # Try trigram first
    if len(tokens) >= 2:
        key = f"{tokens[-2]} {tokens[-1]}"
        if key in trigrams:
            for word, count in sorted(trigrams[key].items(), key=lambda x: x[1], reverse=True)[:max_results]:
                predictions.append({'word': word, 'score': count, 'type': 'trigram'})

    # Fill with bigrams if needed
    if len(predictions) < max_results and len(tokens) >= 1:
        key = tokens[-1]
        if key in bigrams:
            remaining = max_results - len(predictions)
            for word, count in sorted(bigrams[key].items(), key=lambda x: x[1], reverse=True)[:remaining]:
                predictions.append({'word': word, 'score': count, 'type': 'bigram'})

    # Default prediction
    if not predictions:
        predictions = [{'word': 'the', 'score': 1, 'type': 'default'}]

    return {
        'input': text,
        'predictions': predictions
    }


def predict_transformer(text, model_dir='models/asx-gpt2', max_length=50, use_gpu=True):
    """Predict using transformer model"""

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install torch transformers")

    # Setup device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Encode input
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.9
        )

    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        'input': text,
        'output': generated_text,
        'model': model_dir
    }


def main():
    parser = argparse.ArgumentParser(description='Run ASX inference')
    parser.add_argument('text', type=str, help='Input text for prediction')
    parser.add_argument('--type', type=str, default='ngram', choices=['ngram', 'transformer'],
                      help='Model type to use')
    parser.add_argument('--model', type=str, default=None,
                      help='Model directory')
    parser.add_argument('--max-results', type=int, default=5,
                      help='Maximum number of predictions (ngram only)')
    parser.add_argument('--max-length', type=int, default=50,
                      help='Maximum generation length (transformer only)')
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU if available')

    args = parser.parse_args()

    # Set default model directory
    if args.model is None:
        args.model = 'brain' if args.type == 'ngram' else 'models/asx-gpt2'

    # Run prediction
    try:
        if args.type == 'ngram':
            result = predict_ngram(args.text, args.model, args.max_results)
        else:
            result = predict_transformer(args.text, args.model, args.max_length, args.gpu)

        # Output as JSON
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()