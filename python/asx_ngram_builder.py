#!/usr/bin/env python3
"""
ASX N-gram Builder
Build n-gram models from text corpus
"""

import json
import sys
import os
import argparse
from collections import defaultdict
from pathlib import Path


def build_ngrams(corpus_path, output_dir='brain', n_values=[2, 3], min_count=1):
    """Build n-gram models from corpus"""

    print(f"Building n-gram models from {corpus_path}")
    print(f"N-values: {n_values}")

    # Initialize n-gram dictionaries
    ngrams = {n: defaultdict(lambda: defaultdict(int)) for n in n_values}

    # Read corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_tokens = 0

    for line in lines:
        tokens = line.lower().strip().split()
        total_tokens += len(tokens)

        # Build n-grams for each n value
        for n in n_values:
            for i in range(len(tokens) - n + 1):
                # Get context (first n-1 tokens) and target (last token)
                if n == 2:
                    context = tokens[i]
                else:
                    context = ' '.join(tokens[i:i+n-1])

                target = tokens[i+n-1]
                ngrams[n][context][target] += 1

    # Filter by minimum count
    if min_count > 1:
        for n in n_values:
            filtered = {}
            for context, targets in ngrams[n].items():
                filtered_targets = {t: c for t, c in targets.items() if c >= min_count}
                if filtered_targets:
                    filtered[context] = filtered_targets
            ngrams[n] = filtered

    # Save models
    os.makedirs(output_dir, exist_ok=True)

    output_files = {}

    for n in n_values:
        filename = f"{'bigrams' if n == 2 else 'trigrams' if n == 3 else f'{n}grams'}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(dict(ngrams[n]), f, indent=2)

        output_files[f'{n}grams'] = filepath
        print(f"  {n}-grams: {len(ngrams[n])} contexts -> {filepath}")

    print(f"\n✓ N-gram models built successfully!")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Output directory: {output_dir}")

    return {
        'total_tokens': total_tokens,
        'ngrams': {n: len(ngrams[n]) for n in n_values},
        'output_files': output_files
    }


def merge_ngrams(source_dirs, output_dir='brain'):
    """Merge multiple n-gram models"""

    print(f"Merging n-gram models from {len(source_dirs)} sources")

    merged_bigrams = defaultdict(lambda: defaultdict(int))
    merged_trigrams = defaultdict(lambda: defaultdict(int))

    for source_dir in source_dirs:
        # Load bigrams
        bigrams_path = os.path.join(source_dir, 'bigrams.json')
        if os.path.exists(bigrams_path):
            with open(bigrams_path, 'r') as f:
                bigrams = json.load(f)
                for context, targets in bigrams.items():
                    for target, count in targets.items():
                        merged_bigrams[context][target] += count

        # Load trigrams
        trigrams_path = os.path.join(source_dir, 'trigrams.json')
        if os.path.exists(trigrams_path):
            with open(trigrams_path, 'r') as f:
                trigrams = json.load(f)
                for context, targets in trigrams.items():
                    for target, count in targets.items():
                        merged_trigrams[context][target] += count

    # Save merged models
    os.makedirs(output_dir, exist_ok=True)

    bigrams_out = os.path.join(output_dir, 'bigrams.json')
    trigrams_out = os.path.join(output_dir, 'trigrams.json')

    with open(bigrams_out, 'w') as f:
        json.dump(dict(merged_bigrams), f, indent=2)

    with open(trigrams_out, 'w') as f:
        json.dump(dict(merged_trigrams), f, indent=2)

    print(f"\n✓ Models merged successfully!")
    print(f"  Bigrams: {len(merged_bigrams)} contexts")
    print(f"  Trigrams: {len(merged_trigrams)} contexts")
    print(f"  Output: {output_dir}")

    return {
        'bigrams': len(merged_bigrams),
        'trigrams': len(merged_trigrams),
        'output_dir': output_dir
    }


def main():
    parser = argparse.ArgumentParser(description='Build or merge n-gram models')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build n-gram models from corpus')
    build_parser.add_argument('corpus', type=str, help='Path to corpus file')
    build_parser.add_argument('--output', type=str, default='brain',
                             help='Output directory')
    build_parser.add_argument('--n', type=int, nargs='+', default=[2, 3],
                             help='N-gram values to build')
    build_parser.add_argument('--min-count', type=int, default=1,
                             help='Minimum occurrence count')

    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple n-gram models')
    merge_parser.add_argument('sources', type=str, nargs='+',
                             help='Source directories containing n-gram models')
    merge_parser.add_argument('--output', type=str, default='brain',
                             help='Output directory')

    args = parser.parse_args()

    if args.command == 'build':
        if not os.path.exists(args.corpus):
            print(f"Error: Corpus file not found: {args.corpus}")
            sys.exit(1)

        result = build_ngrams(args.corpus, args.output, args.n, args.min_count)
        print("\nResult:")
        print(json.dumps(result, indent=2))

    elif args.command == 'merge':
        for source in args.sources:
            if not os.path.exists(source):
                print(f"Error: Source directory not found: {source}")
                sys.exit(1)

        result = merge_ngrams(args.sources, args.output)
        print("\nResult:")
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()