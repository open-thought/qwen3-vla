#!/usr/bin/env python3
"""
Test script to verify action token diversity in the training data.

Checks that different samples produce different FAST tokens, not always the same sequence.

Usage:
    python dev/test_token_diversity.py                        # No filtering
    python dev/test_token_diversity.py --idle-std-threshold 0.16  # Filter idle frames

Note on thresholds:
    - Idle frames have std(normalized_deltas) ~ 0.152
    - Use --idle-std-threshold 0.16 to filter idle frames effectively
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from collections import Counter
from robotwin_dataset import RoboTwinVLADataset


def main():
    parser = argparse.ArgumentParser(description="Test action token diversity")
    parser.add_argument(
        "--idle-std-threshold",
        type=float,
        default=0.0,
        help="Filter samples where std(normalized_deltas) < threshold. "
             "Idle frames have std ~0.152, so use 0.16 to filter. (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of samples to analyze (default: 2000)",
    )
    args = parser.parse_args()

    print("Loading dataset...")

    # Load dataset with actual normalization
    dataset = RoboTwinVLADataset(
        dataset_root='/mnt/robotwin/dataset',
        norm_stats_path='data/robotwin_norm_stats_h16.json',
        episode_lengths_path='data/robotwin_episode_lengths.json',
        action_horizon=16,
        robot_types=['aloha-agilex'],
        tasks=['place_a2b_left'],
        enable_augmentation=False,
        pad_action_horizon=False,
    )

    print(f'Dataset size: {len(dataset)}')
    if args.idle_std_threshold > 0:
        print(f'Idle std threshold: {args.idle_std_threshold}')
    print()

    # Collect token sequences from multiple samples
    target_samples = min(args.num_samples, len(dataset))
    token_sequences = []
    token_tuples = []  # For counting exact duplicates

    # Tracking for idle filtering
    total_checked = 0
    idle_rejected = 0
    printed_samples = 0

    print(f"Analyzing up to {target_samples} samples...")
    print("=" * 70)

    for i in range(len(dataset)):
        if len(token_tuples) >= target_samples:
            break

        sample = dataset[i]
        total_checked += 1

        # Check idle std threshold if enabled
        if args.idle_std_threshold > 0:
            normalized_deltas = sample['normalized_deltas']
            delta_std = normalized_deltas.std()
            if delta_std < args.idle_std_threshold:
                idle_rejected += 1
                continue

        action_tokens = sample['action_tokens']
        # Remove offset for readability
        tokens_no_offset = tuple(t - 151936 for t in action_tokens)
        token_sequences.append(tokens_no_offset)
        token_tuples.append(tokens_no_offset)

        if printed_samples < 10:
            print(f"Sample {i:3d}: {len(action_tokens):2d} tokens: {list(tokens_no_offset)[:15]}{'...' if len(tokens_no_offset) > 15 else ''}")
            printed_samples += 1

    num_samples = len(token_tuples)

    print()
    print("=" * 70)
    print("Diversity Analysis:")
    print("=" * 70)

    # Show idle filtering stats
    if args.idle_std_threshold > 0:
        print(f"Samples checked: {total_checked}")
        print(f"Idle samples rejected: {idle_rejected} ({idle_rejected/total_checked*100:.1f}%)")
        print(f"Active samples kept: {num_samples} ({num_samples/total_checked*100:.1f}%)")
        print()

    # Count unique sequences
    unique_sequences = set(token_tuples)
    print(f"Total samples analyzed: {num_samples}")
    print(f"Unique token sequences: {len(unique_sequences)}")
    print(f"Duplicate ratio: {1 - len(unique_sequences)/num_samples:.1%}")

    # Count frequency of each unique sequence
    sequence_counts = Counter(token_tuples)
    most_common = sequence_counts.most_common(10)

    print()
    print("Most common sequences:")
    for seq, count in most_common:
        print(f"  Count {count:3d}: {list(seq)[:12]}{'...' if len(seq) > 12 else ''}")

    # Check if first few tokens are always the same
    print()
    print("First token distribution:")
    first_tokens = [seq[0] for seq in token_sequences]
    first_token_counts = Counter(first_tokens)
    for token, count in first_token_counts.most_common(10):
        print(f"  Token {token:4d}: {count:3d} times ({count/num_samples*100:.1f}%)")

    # Check token diversity across positions
    print()
    print("Token diversity by position (unique tokens at each position):")
    max_len = max(len(seq) for seq in token_sequences)
    for pos in range(min(10, max_len)):
        tokens_at_pos = [seq[pos] for seq in token_sequences if len(seq) > pos]
        unique_at_pos = len(set(tokens_at_pos))
        print(f"  Position {pos:2d}: {unique_at_pos:3d} unique tokens out of {len(tokens_at_pos)}")

    # Also check the underlying delta actions
    print()
    print("=" * 70)
    print("Delta Action Analysis:")
    print("=" * 70)

    delta_actions_list = []
    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        # Get the normalized delta actions before tokenization
        # We need to access the raw data
        delta_actions_list.append(i)

    # Sample a few and compare their discretized states
    print()
    print("Discretized state samples (first 6 values):")
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        disc_state = sample['discretized_state'][:6]
        print(f"  Sample {i:3d}: [{', '.join(f'{int(x):3d}' for x in disc_state)}]")

    # Check if the problematic token sequence appears
    problem_tokens = (277, 304, 770, 882, 299, 322, 309, 754, 291, 888, 276)
    matching = [i for i, seq in enumerate(token_tuples) if seq == problem_tokens]
    print()
    if matching:
        print(f"WARNING: The problematic token sequence appears {len(matching)} times in samples: {matching[:20]}")
    else:
        print(f"Good: The problematic token sequence [277, 304, 770, ...] was not found in training data")

    # Check how often similar sequences appear
    print()
    print("Sequences starting with [277, 304]:")
    similar = [i for i, seq in enumerate(token_tuples) if len(seq) >= 2 and seq[0] == 277 and seq[1] == 304]
    print(f"  Found {len(similar)} sequences starting with [277, 304]")


if __name__ == '__main__':
    main()
