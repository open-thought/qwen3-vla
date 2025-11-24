"""
Analyze FAST tokenizer vocabulary usage across RoboTwin dataset.

Tokenizes all normalized delta actions and creates a histogram showing
which vocabulary entries are actually used and how frequently.
"""

import argparse
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from action_tokenizer import ActionTokenizer
from normalization import MultiRobotNormalizer


def analyze_vocab_usage(
    delta_actions_path: str,
    norm_stats_path: str,
    output_dir: str = "dev/vocab_analysis",
):
    """
    Analyze FAST vocabulary usage across the dataset.

    Args:
        delta_actions_path: Path to HDF5 file with extracted delta actions
        norm_stats_path: Path to normalization statistics JSON
        output_dir: Directory to save output plots and statistics
    """
    print("=" * 80)
    print("FAST Tokenizer Vocabulary Usage Analysis")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer and normalizer
    print("\nInitializing tokenizer and normalizer...")
    tokenizer = ActionTokenizer()
    normalizer = MultiRobotNormalizer(norm_stats_path)

    vocab_start, vocab_end = tokenizer.get_token_range()
    vocab_size = tokenizer.VOCAB_SIZE
    print(f"  FAST vocab size: {vocab_size}")
    print(f"  Token range: [{vocab_start}, {vocab_end}]")

    # Counter for token occurrences
    token_counter = Counter()
    total_tokens = 0
    total_samples = 0

    # Process all robot types
    print(f"\nLoading delta actions from {delta_actions_path}...")
    with h5py.File(delta_actions_path, "r") as f:
        robot_types = list(f.keys())
        print(f"Found {len(robot_types)} robot types: {robot_types}")

        for robot_type in robot_types:
            print(f"\nProcessing {robot_type}...")

            # Load delta actions
            deltas = f[f"{robot_type}/delta_actions"][:]  # (N, action_horizon, 2*dof)
            num_samples = deltas.shape[0]
            action_horizon = deltas.shape[1]
            action_dim = deltas.shape[2]

            print(f"  Samples: {num_samples}")
            print(f"  Action horizon: {action_horizon}")
            print(f"  Action dim: {action_dim}")

            # Normalize delta actions
            print(f"  Normalizing delta actions...")
            normalized_deltas = normalizer.normalize_delta_actions(deltas, robot_type)

            # Tokenize in batches to save memory
            batch_size = 1000
            num_batches = (num_samples + batch_size - 1) // batch_size

            print(f"  Tokenizing {num_samples} samples in {num_batches} batches...")
            for batch_idx in tqdm(range(num_batches), desc=f"  {robot_type}"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)

                batch_deltas = normalized_deltas[start_idx:end_idx]

                # Encode batch
                token_sequences = tokenizer.encode(batch_deltas, return_torch=False)

                # Count tokens
                for token_seq in token_sequences:
                    token_counter.update(token_seq)
                    total_tokens += len(token_seq)

                total_samples += len(batch_deltas)

    print(f"\n" + "=" * 80)
    print(f"Tokenization complete!")
    print(f"  Total samples processed: {total_samples:,}")
    print(f"  Total tokens generated: {total_tokens:,}")
    print(f"  Average tokens per sample: {total_tokens / total_samples:.2f}")
    print(f"  Unique tokens used: {len(token_counter):,} / {vocab_size:,} ({100 * len(token_counter) / vocab_size:.1f}%)")

    # Analyze vocabulary usage
    print(f"\n" + "=" * 80)
    print("Vocabulary Usage Statistics:")
    print("=" * 80)

    # Sort tokens by frequency
    sorted_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)

    # Top 20 most frequent tokens
    print("\nTop 20 most frequent tokens:")
    for i, (token, count) in enumerate(sorted_tokens[:20], 1):
        percentage = 100 * count / total_tokens
        print(f"  {i:2d}. Token {token:6d}: {count:10,} ({percentage:5.2f}%)")

    # Coverage analysis
    cumulative_coverage = []
    cumulative_count = 0
    for token, count in sorted_tokens:
        cumulative_count += count
        cumulative_coverage.append((token, cumulative_count / total_tokens))

    # Find tokens covering 50%, 90%, 99% of usage
    for threshold in [0.50, 0.90, 0.99]:
        for i, (token, coverage) in enumerate(cumulative_coverage):
            if coverage >= threshold:
                print(f"\n{int(threshold * 100)}% of tokens covered by top {i + 1} vocabulary entries")
                break

    # Create visualizations
    print(f"\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)

    # 1. Histogram of token usage (sorted by frequency)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Token frequency distribution (sorted)
    ax = axes[0, 0]
    token_counts = [count for _, count in sorted_tokens]
    ax.bar(range(len(token_counts)), token_counts)
    ax.set_xlabel("Token Rank (sorted by frequency)")
    ax.set_ylabel("Count")
    ax.set_title(f"Token Frequency Distribution (Sorted)\n{len(token_counter)} unique tokens used")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 2: Top 50 tokens
    ax = axes[0, 1]
    top_n = min(50, len(sorted_tokens))
    top_tokens = [token for token, _ in sorted_tokens[:top_n]]
    top_counts = [count for _, count in sorted_tokens[:top_n]]
    ax.bar(range(top_n), top_counts)
    ax.set_xlabel("Token Rank")
    ax.set_ylabel("Count")
    ax.set_title(f"Top {top_n} Most Frequent Tokens")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative coverage
    ax = axes[1, 0]
    coverage_percentages = [cov * 100 for _, cov in cumulative_coverage]
    ax.plot(range(len(coverage_percentages)), coverage_percentages, linewidth=2)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=99, color='yellow', linestyle='--', alpha=0.5, label='99%')
    ax.set_xlabel("Number of Top Tokens")
    ax.set_ylabel("Cumulative Coverage (%)")
    ax.set_title("Cumulative Token Coverage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(1000, len(cumulative_coverage)))

    # Plot 4: Vocabulary coverage across entire range
    ax = axes[1, 1]
    # Create bins for vocabulary (e.g., 64 bins across 2048 tokens)
    num_bins = 64
    bin_size = vocab_size // num_bins
    bin_counts = [0] * num_bins

    for token, count in token_counter.items():
        # Map token to bin (remove vocab offset)
        token_relative = token - vocab_start
        bin_idx = min(token_relative // bin_size, num_bins - 1)
        bin_counts[bin_idx] += count

    ax.bar(range(num_bins), bin_counts)
    ax.set_xlabel(f"Vocabulary Bin ({bin_size} tokens per bin)")
    ax.set_ylabel("Total Count")
    ax.set_title(f"Token Usage Across Vocabulary Range\n({num_bins} bins, {vocab_size} total tokens)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_path / "fast_vocab_usage.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_file}")

    # Save detailed statistics
    stats_file = output_path / "vocab_usage_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("FAST Tokenizer Vocabulary Usage Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {delta_actions_path}\n")
        f.write(f"Normalization stats: {norm_stats_path}\n\n")
        f.write(f"Total samples processed: {total_samples:,}\n")
        f.write(f"Total tokens generated: {total_tokens:,}\n")
        f.write(f"Average tokens per sample: {total_tokens / total_samples:.2f}\n")
        f.write(f"Vocabulary size: {vocab_size:,}\n")
        f.write(f"Unique tokens used: {len(token_counter):,} ({100 * len(token_counter) / vocab_size:.1f}%)\n")
        f.write(f"Unused tokens: {vocab_size - len(token_counter):,}\n\n")

        f.write("Top 100 Most Frequent Tokens:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Token ID':<10} {'Count':<15} {'Percentage':<12}\n")
        f.write("-" * 80 + "\n")
        for i, (token, count) in enumerate(sorted_tokens[:100], 1):
            percentage = 100 * count / total_tokens
            f.write(f"{i:<6} {token:<10} {count:<15,} {percentage:5.2f}%\n")

    print(f"✓ Saved statistics to {stats_file}")

    # Save raw token counts
    counts_file = output_path / "token_counts.txt"
    with open(counts_file, 'w') as f:
        f.write("Token ID,Count\n")
        for token, count in sorted_tokens:
            f.write(f"{token},{count}\n")

    print(f"✓ Saved raw token counts to {counts_file}")

    print(f"\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FAST tokenizer vocabulary usage across RoboTwin dataset"
    )
    parser.add_argument(
        "--delta-actions",
        type=str,
        default="data/robotwin_delta_actions_h16.hdf5",
        help="Path to HDF5 file with extracted delta actions",
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="data/robotwin_norm_stats_h16.json",
        help="Path to normalization statistics JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dev/vocab_analysis",
        help="Output directory for plots and statistics",
    )

    args = parser.parse_args()

    analyze_vocab_usage(
        delta_actions_path=args.delta_actions,
        norm_stats_path=args.norm_stats,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
