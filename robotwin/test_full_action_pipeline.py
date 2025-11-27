"""
Full pipeline test for action tokenization with grippers.

Tests the complete roundtrip for both FAST and BinTokenizer:
1. Dataset extracts delta joints + absolute grippers
2. Normalizes and concatenates to 14-dim actions
3. Tokenizes the normalized actions (FAST or Bin)
4. Detokenizes back to normalized actions
5. Splits and denormalizes joints and grippers separately
6. Verifies reconstruction matches original (within tokenizer compression tolerance)

Supports testing with symmetric normalization (n_bins=257) for exact zero reconstruction.
"""

import numpy as np
import torch

from robotwin_dataset import RoboTwinVLADataset
from action_tokenizer import create_action_tokenizer, FASTTokenizer, BinTokenizer
from normalization import MultiRobotNormalizer


def test_full_action_pipeline(
    tokenizer_type: str = "fast",
    n_bins: int = 256,
    symmetric_delta_norm: bool = False,
):
    """Test the complete action tokenization pipeline including grippers.

    Args:
        tokenizer_type: "fast" or "bin"
        n_bins: Number of bins for BinTokenizer (256 legacy, 257 for exact zero)
        symmetric_delta_norm: Use symmetric normalization for delta actions
    """
    print("=" * 80)
    mode_str = "symmetric" if symmetric_delta_norm else "legacy"
    print(f"FULL ACTION PIPELINE TEST - {tokenizer_type.upper()} Tokenizer ({n_bins} bins, {mode_str})")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading dataset...")
    dataset = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/robotwin_norm_stats_h16.json",
        episode_lengths_path="data/robotwin_episode_lengths.json",
        action_horizon=16,
        image_size=(320, 240),
        tasks=["beat_block_hammer"],
        robot_types=["aloha-agilex"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=False,
        tokenizer_type=tokenizer_type,
        n_bins=n_bins,
        symmetric_delta_norm=symmetric_delta_norm,
    )
    print(f"   Dataset size: {len(dataset)}")

    # Get a sample
    print("\n2. Getting sample from dataset...")
    sample = dataset[0]

    robot_type = sample["robot_type"]
    action_horizon = sample["actual_action_horizon"]

    print(f"   Robot type: {robot_type}")
    print(f"   Action horizon: {action_horizon}")

    # Check the normalized_deltas shape (should be 14 = 12 joints + 2 grippers)
    normalized_deltas = sample["normalized_deltas"]
    print(f"\n3. Checking normalized_deltas from dataset...")
    print(f"   Shape: {normalized_deltas.shape}")
    print(f"   Expected: ({action_horizon}, 14)")
    assert normalized_deltas.shape == (action_horizon, 14), \
        f"Expected shape ({action_horizon}, 14), got {normalized_deltas.shape}"
    print(f"   Range: [{normalized_deltas.min():.4f}, {normalized_deltas.max():.4f}]")

    # Split into joints and grippers
    joint_dim = 12  # 2 * 6 DoF
    original_normalized_joints = normalized_deltas[:, :joint_dim]
    original_normalized_grippers = normalized_deltas[:, joint_dim:]

    print(f"   Joint deltas (normalized) shape: {original_normalized_joints.shape}")
    print(f"   Gripper values (normalized) shape: {original_normalized_grippers.shape}")

    # Get original raw values for comparison
    delta_joints = sample["delta_joints"]
    future_grippers = sample["future_grippers"]
    print(f"\n4. Original raw values from dataset...")
    print(f"   Delta joints shape: {delta_joints.shape}, range: [{delta_joints.min():.4f}, {delta_joints.max():.4f}]")
    print(f"   Future grippers shape: {future_grippers.shape}, range: [{future_grippers.min():.4f}, {future_grippers.max():.4f}]")

    # Get action tokens from dataset
    action_tokens = sample["action_tokens"]
    print(f"\n5. Action tokens from dataset...")
    print(f"   Number of tokens: {len(action_tokens)}")
    print(f"   Tokens: {action_tokens}")

    # Now simulate the evaluation pipeline (as in qwen3_vla_policy.py)
    print("\n6. Simulating evaluation pipeline (decode tokens)...")

    tokenizer = create_action_tokenizer(tokenizer_type, n_bins=n_bins)
    normalizer = MultiRobotNormalizer("data/robotwin_norm_stats_h16.json", symmetric_delta_norm=symmetric_delta_norm)

    # Decode tokens back to normalized actions
    # action_dim = 14 (12 joints + 2 grippers)
    action_dim = 14
    decoded_normalized = tokenizer.decode(
        [action_tokens],
        action_horizon=action_horizon,
        action_dim=action_dim
    )[0]  # Remove batch dim

    print(f"   Decoded normalized shape: {decoded_normalized.shape}")
    print(f"   Decoded normalized range: [{decoded_normalized.min():.4f}, {decoded_normalized.max():.4f}]")

    # Split decoded into joints and grippers (as done in qwen3_vla_policy.py)
    decoded_normalized_joints = decoded_normalized[:, :joint_dim]
    decoded_normalized_grippers = decoded_normalized[:, joint_dim:]

    print(f"\n7. Split decoded normalized actions...")
    print(f"   Decoded joint deltas shape: {decoded_normalized_joints.shape}")
    print(f"   Decoded gripper values shape: {decoded_normalized_grippers.shape}")

    # Denormalize separately (as done in qwen3_vla_policy.py)
    decoded_delta_joints = normalizer.denormalize_delta_actions(
        decoded_normalized_joints, robot_type
    )
    decoded_future_grippers = normalizer.denormalize_grippers(
        decoded_normalized_grippers, robot_type
    )

    print(f"\n8. Denormalized decoded actions...")
    print(f"   Decoded delta joints range: [{decoded_delta_joints.min():.4f}, {decoded_delta_joints.max():.4f}]")
    print(f"   Decoded future grippers range: [{decoded_future_grippers.min():.4f}, {decoded_future_grippers.max():.4f}]")

    # Compare with original
    print("\n9. Comparing original vs decoded...")

    joint_mae = np.abs(delta_joints - decoded_delta_joints).mean()
    joint_max_error = np.abs(delta_joints - decoded_delta_joints).max()

    gripper_mae = np.abs(future_grippers - decoded_future_grippers).mean()
    gripper_max_error = np.abs(future_grippers - decoded_future_grippers).max()

    print(f"   Joint deltas:")
    print(f"     MAE: {joint_mae:.6f}")
    print(f"     Max error: {joint_max_error:.6f}")

    print(f"   Future grippers:")
    print(f"     MAE: {gripper_mae:.6f}")
    print(f"     Max error: {gripper_max_error:.6f}")

    # Print first few timesteps for visual comparison
    print(f"\n10. Sample comparison (first 3 timesteps)...")
    print("    Original delta joints vs Decoded delta joints:")
    for t in range(min(3, action_horizon)):
        orig = delta_joints[t]
        dec = decoded_delta_joints[t]
        print(f"    t={t}: orig=[{', '.join(f'{x:7.4f}' for x in orig[:6])}...]")
        print(f"         dec =[{', '.join(f'{x:7.4f}' for x in dec[:6])}...]")

    print("\n    Original grippers vs Decoded grippers:")
    for t in range(min(3, action_horizon)):
        orig = future_grippers[t]
        dec = decoded_future_grippers[t]
        print(f"    t={t}: orig=[{orig[0]:.4f}, {orig[1]:.4f}] dec=[{dec[0]:.4f}, {dec[1]:.4f}]")

    # Verify grippers are actually varying (not constant)
    print("\n11. Verifying grippers have variation...")
    gripper_std = future_grippers.std()
    decoded_gripper_std = decoded_future_grippers.std()
    print(f"    Original gripper std: {gripper_std:.6f}")
    print(f"    Decoded gripper std: {decoded_gripper_std:.6f}")

    # Check if grippers span a reasonable range
    gripper_range = future_grippers.max() - future_grippers.min()
    print(f"    Original gripper range: {gripper_range:.4f}")

    # Assertions
    print("\n12. Running assertions...")

    # Set error thresholds based on tokenizer type and normalization mode
    # FAST has compression loss, Bin is more precise (256 bins = ~0.008 bin width)
    if tokenizer_type == "bin":
        if symmetric_delta_norm and n_bins == 257:
            # Symmetric + 257 bins should have minimal error
            joint_threshold = 0.015
            gripper_threshold = 0.015
        else:
            joint_threshold = 0.02  # BinTokenizer is more precise
            gripper_threshold = 0.02
    else:
        joint_threshold = 0.1  # FAST compression has more loss
        gripper_threshold = 0.1

    # Joint errors should be small relative to typical delta magnitudes
    assert joint_mae < joint_threshold, f"Joint MAE too large: {joint_mae} (threshold: {joint_threshold})"
    print(f"    ✓ Joint MAE < {joint_threshold}")

    # Gripper errors should be very small since they're in [0, 1]
    assert gripper_mae < gripper_threshold, f"Gripper MAE too large: {gripper_mae} (threshold: {gripper_threshold})"
    print(f"    ✓ Gripper MAE < {gripper_threshold}")

    # Verify shapes are correct
    assert decoded_delta_joints.shape == delta_joints.shape, "Joint shape mismatch"
    print(f"    ✓ Joint shapes match")

    assert decoded_future_grippers.shape == future_grippers.shape, "Gripper shape mismatch"
    print(f"    ✓ Gripper shapes match")

    dataset.close()

    print("\n" + "=" * 80)
    print(f"✅ FULL ACTION PIPELINE TEST PASSED ({tokenizer_type.upper()}, {n_bins} bins, {mode_str})")
    print("=" * 80)

    return {
        "tokenizer_type": tokenizer_type,
        "n_bins": n_bins,
        "symmetric_delta_norm": symmetric_delta_norm,
        "joint_mae": joint_mae,
        "joint_max_error": joint_max_error,
        "gripper_mae": gripper_mae,
        "gripper_max_error": gripper_max_error,
    }


def test_multiple_samples(
    tokenizer_type: str = "fast",
    n_bins: int = 256,
    symmetric_delta_norm: bool = False,
):
    """Test pipeline on multiple samples to get statistics.

    Args:
        tokenizer_type: "fast" or "bin"
        n_bins: Number of bins for BinTokenizer (256 legacy, 257 for exact zero)
        symmetric_delta_norm: Use symmetric normalization for delta actions
    """
    mode_str = "symmetric" if symmetric_delta_norm else "legacy"
    print("\n" + "=" * 80)
    print(f"TESTING MULTIPLE SAMPLES - {tokenizer_type.upper()} Tokenizer ({n_bins} bins, {mode_str})")
    print("=" * 80)

    dataset = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/robotwin_norm_stats_h8.json",
        episode_lengths_path="data/robotwin_episode_lengths.json",
        valid_timesteps_path="data/robotwin_valid_timesteps_h8.json",
        action_horizon=8,
        image_size=(320, 240),
        tasks=["beat_block_hammer"],
        robot_types=["aloha-agilex"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=False,
        tokenizer_type=tokenizer_type,
        n_bins=n_bins,
        symmetric_delta_norm=symmetric_delta_norm,
    )

    tokenizer = create_action_tokenizer(tokenizer_type, n_bins=n_bins)
    normalizer = MultiRobotNormalizer("data/robotwin_norm_stats_h16.json", symmetric_delta_norm=symmetric_delta_norm)

    joint_maes = []
    gripper_maes = []
    num_samples = min(50, len(dataset))

    print(f"\nTesting {num_samples} samples...")

    for i in range(num_samples):
        sample = dataset[i]
        robot_type = sample["robot_type"]
        action_horizon = sample["actual_action_horizon"]

        # Original values
        delta_joints = sample["delta_joints"]
        future_grippers = sample["future_grippers"]
        action_tokens = sample["action_tokens"]
        print(f"{i}: {[x-tokenizer.VOCAB_OFFSET for x in action_tokens]} ({len(action_tokens)} tokens)") 

        # Decode
        decoded_normalized = tokenizer.decode(
            [action_tokens],
            action_horizon=action_horizon,
            action_dim=14
        )[0]

        # Split and denormalize
        decoded_delta_joints = normalizer.denormalize_delta_actions(
            decoded_normalized[:, :12], robot_type
        )
        decoded_future_grippers = normalizer.denormalize_grippers(
            decoded_normalized[:, 12:], robot_type
        )

        # Compute errors
        joint_mae = np.abs(delta_joints - decoded_delta_joints).mean()
        gripper_mae = np.abs(future_grippers - decoded_future_grippers).mean()

        joint_maes.append(joint_mae)
        gripper_maes.append(gripper_mae)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples...")

    dataset.close()

    print(f"\nResults over {num_samples} samples ({tokenizer_type.upper()}, {n_bins} bins, {mode_str}):")
    print(f"  Joint delta MAE:  mean={np.mean(joint_maes):.6f}, std={np.std(joint_maes):.6f}, max={np.max(joint_maes):.6f}")
    print(f"  Gripper MAE:      mean={np.mean(gripper_maes):.6f}, std={np.std(gripper_maes):.6f}, max={np.max(gripper_maes):.6f}")

    print("\n" + "=" * 80)
    print(f"✅ MULTIPLE SAMPLES TEST PASSED ({tokenizer_type.upper()}, {n_bins} bins, {mode_str})")
    print("=" * 80)

    return {
        "tokenizer_type": tokenizer_type,
        "n_bins": n_bins,
        "symmetric_delta_norm": symmetric_delta_norm,
        "joint_mae_mean": np.mean(joint_maes),
        "joint_mae_std": np.std(joint_maes),
        "gripper_mae_mean": np.mean(gripper_maes),
        "gripper_mae_std": np.std(gripper_maes),
    }


def test_both_tokenizers():
    """Test both FAST and Bin tokenizers and compare results."""
    print("\n" + "=" * 80)
    print("COMPARING FAST VS BIN TOKENIZERS")
    print("=" * 80)

    results = {}

    # Test FAST tokenizer
    print("\n>>> Testing FAST tokenizer <<<")
    results["fast"] = test_full_action_pipeline("fast")

    # Test BIN tokenizer (legacy: 256 bins, asymmetric)
    print("\n>>> Testing BIN tokenizer (legacy: 256 bins) <<<")
    results["bin_legacy"] = test_full_action_pipeline("bin", n_bins=256, symmetric_delta_norm=False)

    # Test BIN tokenizer (new: 257 bins, symmetric)
    print("\n>>> Testing BIN tokenizer (symmetric: 257 bins) <<<")
    results["bin_symmetric"] = test_full_action_pipeline("bin", n_bins=257, symmetric_delta_norm=True)

    # Test multiple samples with all
    print("\n>>> Testing multiple samples with FAST <<<")
    fast_stats = test_multiple_samples("fast")

    print("\n>>> Testing multiple samples with BIN (legacy) <<<")
    bin_legacy_stats = test_multiple_samples("bin", n_bins=256, symmetric_delta_norm=False)

    print("\n>>> Testing multiple samples with BIN (symmetric) <<<")
    bin_symmetric_stats = test_multiple_samples("bin", n_bins=257, symmetric_delta_norm=True)

    # Print comparison summary
    print("\n" + "=" * 80)
    print("TOKENIZER COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nSingle sample results:")
    print(f"  FAST:          Joint MAE={results['fast']['joint_mae']:.6f}, Gripper MAE={results['fast']['gripper_mae']:.6f}")
    print(f"  BIN (legacy):  Joint MAE={results['bin_legacy']['joint_mae']:.6f}, Gripper MAE={results['bin_legacy']['gripper_mae']:.6f}")
    print(f"  BIN (symm):    Joint MAE={results['bin_symmetric']['joint_mae']:.6f}, Gripper MAE={results['bin_symmetric']['gripper_mae']:.6f}")

    print(f"\nMultiple samples results (mean ± std):")
    print(f"  FAST:          Joint MAE={fast_stats['joint_mae_mean']:.6f}±{fast_stats['joint_mae_std']:.6f}, "
          f"Gripper MAE={fast_stats['gripper_mae_mean']:.6f}±{fast_stats['gripper_mae_std']:.6f}")
    print(f"  BIN (legacy):  Joint MAE={bin_legacy_stats['joint_mae_mean']:.6f}±{bin_legacy_stats['joint_mae_std']:.6f}, "
          f"Gripper MAE={bin_legacy_stats['gripper_mae_mean']:.6f}±{bin_legacy_stats['gripper_mae_std']:.6f}")
    print(f"  BIN (symm):    Joint MAE={bin_symmetric_stats['joint_mae_mean']:.6f}±{bin_symmetric_stats['joint_mae_std']:.6f}, "
          f"Gripper MAE={bin_symmetric_stats['gripper_mae_mean']:.6f}±{bin_symmetric_stats['gripper_mae_std']:.6f}")

    print("\n" + "=" * 80)
    print("✅ ALL TOKENIZER TESTS PASSED")
    print("=" * 80)


def test_zero_reconstruction():
    """Test that zeros are reconstructed exactly with symmetric normalization."""
    print("\n" + "=" * 80)
    print("TESTING ZERO RECONSTRUCTION (symmetric normalization)")
    print("=" * 80)

    from normalization import MultiRobotNormalizer
    from action_tokenizer import BinTokenizer

    # Test legacy (should NOT be exact)
    print("\n1. Legacy mode (256 bins, asymmetric):")
    norm_legacy = MultiRobotNormalizer("data/robotwin_norm_stats_h16.json", symmetric_delta_norm=False)
    tok_legacy = BinTokenizer(n_bins=256)

    zeros = np.zeros((4, 12), dtype=np.float32)
    normalized = norm_legacy.normalize_delta_actions(zeros, "aloha-agilex")
    tokens = tok_legacy.encode(normalized[None, :])
    decoded_norm = tok_legacy.decode(tokens, action_horizon=4, action_dim=12)[0]
    decoded = norm_legacy.denormalize_delta_actions(decoded_norm, "aloha-agilex")

    legacy_max_error = np.abs(decoded).max()
    print(f"   Max error for zeros: {legacy_max_error:.6f}")
    print(f"   Decoded first row: {decoded[0, :6]}")

    # Test symmetric (should be exact)
    print("\n2. Symmetric mode (257 bins, symmetric):")
    norm_sym = MultiRobotNormalizer("data/robotwin_norm_stats_h16.json", symmetric_delta_norm=True)
    tok_sym = BinTokenizer(n_bins=257)

    normalized = norm_sym.normalize_delta_actions(zeros, "aloha-agilex")
    tokens = tok_sym.encode(normalized[None, :])
    decoded_norm = tok_sym.decode(tokens, action_horizon=4, action_dim=12)[0]
    decoded = norm_sym.denormalize_delta_actions(decoded_norm, "aloha-agilex")

    sym_max_error = np.abs(decoded).max()
    print(f"   Max error for zeros: {sym_max_error:.6f}")
    print(f"   Decoded first row: {decoded[0, :6]}")

    # Assertions
    print("\n3. Assertions:")
    assert sym_max_error < 1e-9, f"Symmetric mode should have zero error, got {sym_max_error}"
    print(f"   ✓ Symmetric mode has zero error")

    assert legacy_max_error > 0.0001, f"Legacy mode should have non-zero error"
    print(f"   ✓ Legacy mode has non-zero error (expected)")

    print("\n" + "=" * 80)
    print("✅ ZERO RECONSTRUCTION TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test action tokenization pipeline")
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default="both",
        choices=["fast", "bin", "bin-symmetric", "both"],
        help="Which tokenizer to test (default: both)"
    )
    parser.add_argument(
        "--test-zeros",
        action="store_true",
        help="Run zero reconstruction test"
    )
    args = parser.parse_args()

    if args.test_zeros:
        test_zero_reconstruction()
    elif args.tokenizer == "both":
        test_both_tokenizers()
    elif args.tokenizer == "bin-symmetric":
        test_full_action_pipeline("bin", n_bins=257, symmetric_delta_norm=True)
        test_multiple_samples("bin", n_bins=257, symmetric_delta_norm=True)
    else:
        test_full_action_pipeline(args.tokenizer)
        test_multiple_samples(args.tokenizer)
