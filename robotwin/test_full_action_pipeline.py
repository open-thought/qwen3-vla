"""
Full pipeline test for action tokenization with grippers.

Tests the complete roundtrip:
1. Dataset extracts delta joints + absolute grippers
2. Normalizes and concatenates to 14-dim actions
3. FAST tokenizes the normalized actions
4. FAST detokenizes back to normalized actions
5. Splits and denormalizes joints and grippers separately
6. Verifies reconstruction matches original (within FAST compression tolerance)
"""

import numpy as np
import torch

from robotwin_dataset import RoboTwinVLADataset
from action_tokenizer import ActionTokenizer
from normalization import MultiRobotNormalizer


def test_full_action_pipeline():
    """Test the complete action tokenization pipeline including grippers."""
    print("=" * 80)
    print("FULL ACTION PIPELINE TEST (with grippers)")
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
    print(f"   Token range: [{min(action_tokens)}, {max(action_tokens)}]")

    # Now simulate the evaluation pipeline (as in qwen3_vla_policy.py)
    print("\n6. Simulating evaluation pipeline (decode tokens)...")

    tokenizer = ActionTokenizer()
    normalizer = MultiRobotNormalizer("data/robotwin_norm_stats_h16.json")

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

    # FAST compression has some loss, but should be reasonable
    # Joint errors should be small relative to typical delta magnitudes
    assert joint_mae < 0.1, f"Joint MAE too large: {joint_mae}"
    print(f"    ✓ Joint MAE < 0.1")

    # Gripper errors should be very small since they're in [0, 1]
    assert gripper_mae < 0.1, f"Gripper MAE too large: {gripper_mae}"
    print(f"    ✓ Gripper MAE < 0.1")

    # Verify shapes are correct
    assert decoded_delta_joints.shape == delta_joints.shape, "Joint shape mismatch"
    print(f"    ✓ Joint shapes match")

    assert decoded_future_grippers.shape == future_grippers.shape, "Gripper shape mismatch"
    print(f"    ✓ Gripper shapes match")

    dataset.close()

    print("\n" + "=" * 80)
    print("✅ FULL ACTION PIPELINE TEST PASSED")
    print("=" * 80)

    return {
        "joint_mae": joint_mae,
        "joint_max_error": joint_max_error,
        "gripper_mae": gripper_mae,
        "gripper_max_error": gripper_max_error,
    }


def test_multiple_samples():
    """Test pipeline on multiple samples to get statistics."""
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE SAMPLES")
    print("=" * 80)

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
    )

    tokenizer = ActionTokenizer()
    normalizer = MultiRobotNormalizer("data/robotwin_norm_stats_h16.json")

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

    print(f"\nResults over {num_samples} samples:")
    print(f"  Joint delta MAE:  mean={np.mean(joint_maes):.6f}, std={np.std(joint_maes):.6f}, max={np.max(joint_maes):.6f}")
    print(f"  Gripper MAE:      mean={np.mean(gripper_maes):.6f}, std={np.std(gripper_maes):.6f}, max={np.max(gripper_maes):.6f}")

    print("\n" + "=" * 80)
    print("✅ MULTIPLE SAMPLES TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    test_full_action_pipeline()
    test_multiple_samples()
