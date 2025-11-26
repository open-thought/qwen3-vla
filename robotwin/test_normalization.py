"""Test script for normalization utilities."""

import numpy as np

from normalization import MultiRobotNormalizer, discretize_normalized_values, undiscretize_to_normalized


def test_normalization_roundtrip():
    """Test normalization and denormalization round-trip with synthetic data."""
    print("Testing normalization utilities...")
    print("=" * 60)

    # Load normalizer
    normalizer = MultiRobotNormalizer("data/robotwin_norm_stats_h16.json")
    print(f"\nLoaded normalizer for robot types: {normalizer.robot_types}")

    for robot_type in normalizer.robot_types:
        print(f"\n{'='*60}")
        print(f"Testing {robot_type}")
        print(f"{'='*60}")

        # Get metadata
        metadata = normalizer.get_robot_metadata(robot_type)
        dof = metadata['dof']
        joint_dim = 2 * dof
        print(f"\nMetadata:")
        print(f"  DoF per arm: {dof}")
        print(f"  Action horizon: {metadata['action_horizon']}")
        print(f"  Num samples: {metadata['num_samples']}")

        # Create synthetic test data within the q01-q99 range
        np.random.seed(42)

        # Get stats to create reasonable test values
        state_q01 = normalizer.stats[robot_type]["state"]["q01"]
        state_q99 = normalizer.stats[robot_type]["state"]["q99"]
        delta_q01 = normalizer.stats[robot_type]["delta_actions"]["q01"]
        delta_q99 = normalizer.stats[robot_type]["delta_actions"]["q99"]

        # Generate states within 80% of the q01-q99 range (to avoid edge effects)
        state_range = (state_q99 - state_q01) * 0.4
        state_center = (state_q99 + state_q01) / 2
        states = (np.random.rand(100, joint_dim).astype(np.float32) - 0.5) * state_range + state_center

        # Generate deltas within 80% of the q01-q99 range
        delta_range = (delta_q99 - delta_q01) * 0.4
        delta_center = (delta_q99 + delta_q01) / 2
        deltas = (np.random.rand(100, joint_dim).astype(np.float32) - 0.5) * delta_range + delta_center

        # Grippers are in [0, 1] range
        grippers = np.random.rand(100, 2).astype(np.float32)

        print(f"\nSynthetic data ranges:")
        print(f"  States: [{states.min():.4f}, {states.max():.4f}]")
        print(f"  Deltas: [{deltas.min():.4f}, {deltas.max():.4f}]")
        print(f"  Grippers: [{grippers.min():.4f}, {grippers.max():.4f}]")

        # Test state normalization
        normalized_states = normalizer.normalize_state(states, robot_type)
        print(f"\nNormalized states range: [{normalized_states.min():.4f}, {normalized_states.max():.4f}]")
        # Values should be mostly in [-1, 1] since we generated data within q01-q99
        assert normalized_states.min() >= -1.5 and normalized_states.max() <= 1.5, \
            f"Normalized states unexpectedly outside [-1.5, 1.5]: [{normalized_states.min():.4f}, {normalized_states.max():.4f}]"

        # Test state denormalization (round-trip)
        denormalized_states = normalizer.denormalize_state(normalized_states, robot_type)
        state_error = np.abs(states - denormalized_states).mean()
        print(f"State round-trip error: {state_error:.6f}")
        assert state_error < 1e-5, f"State round-trip error too large: {state_error}"

        # Test delta action normalization
        normalized_deltas = normalizer.normalize_delta_actions(deltas, robot_type)
        print(f"\nNormalized deltas range: [{normalized_deltas.min():.4f}, {normalized_deltas.max():.4f}]")
        assert normalized_deltas.min() >= -1.5 and normalized_deltas.max() <= 1.5, \
            f"Normalized deltas unexpectedly outside [-1.5, 1.5]: [{normalized_deltas.min():.4f}, {normalized_deltas.max():.4f}]"

        # Test delta action denormalization (round-trip)
        denormalized_deltas = normalizer.denormalize_delta_actions(normalized_deltas, robot_type)
        delta_error = np.abs(deltas - denormalized_deltas).mean()
        print(f"Delta round-trip error: {delta_error:.6f}")
        assert delta_error < 1e-5, f"Delta round-trip error too large: {delta_error}"

        # Test gripper normalization
        normalized_grippers = normalizer.normalize_grippers(grippers, robot_type)
        print(f"\nNormalized grippers range: [{normalized_grippers.min():.4f}, {normalized_grippers.max():.4f}]")
        # Grippers are [0, 1] which maps to [-1, 1]
        assert normalized_grippers.min() >= -1.1 and normalized_grippers.max() <= 1.1, \
            f"Normalized grippers unexpectedly outside [-1.1, 1.1]: [{normalized_grippers.min():.4f}, {normalized_grippers.max():.4f}]"

        # Test gripper denormalization (round-trip)
        denormalized_grippers = normalizer.denormalize_grippers(normalized_grippers, robot_type)
        gripper_error = np.abs(grippers - denormalized_grippers).mean()
        print(f"Gripper round-trip error: {gripper_error:.6f}")
        assert gripper_error < 1e-5, f"Gripper round-trip error too large: {gripper_error}"

        # Test discretization (clamps to [-1, 1] before discretizing)
        # Use values that are within [-1, 1] for this test
        clamped_normalized = np.clip(normalized_states, -1.0, 1.0)
        discretized = discretize_normalized_values(clamped_normalized)
        print(f"\nDiscretized state range: [{discretized.min()}, {discretized.max()}]")
        assert discretized.min() >= 0 and discretized.max() <= 255, \
            "Discretized values outside [0, 255] range!"

        # Test undiscretization
        undiscretized = undiscretize_to_normalized(discretized)
        print(f"Undiscretized state range: [{undiscretized.min():.4f}, {undiscretized.max():.4f}]")
        discretization_error = np.abs(clamped_normalized - undiscretized).mean()
        print(f"Discretization round-trip error: {discretization_error:.6f}")
        # Note: Some error is expected due to quantization (256 bins)
        assert discretization_error < 0.01, f"Discretization error too large: {discretization_error}"

        print(f"\n✓ All tests passed for {robot_type}")

    print("\n" + "=" * 60)
    print("✓ All normalization tests passed successfully!")
    print("=" * 60)


def test_combined_action_roundtrip():
    """
    Test the combined action format used in training/evaluation.

    The action format is (action_horizon, 14) where:
    - First 12 dims: normalized joint deltas (using delta_actions stats)
    - Last 2 dims: normalized absolute gripper values (using grippers stats)

    This tests the full roundtrip as used in qwen3_vla_policy.py.
    """
    print("\n" + "=" * 60)
    print("Testing combined action format (joints + grippers) roundtrip")
    print("=" * 60)

    # Load normalizer
    normalizer = MultiRobotNormalizer("data/robotwin_norm_stats_h16.json")

    for robot_type in normalizer.robot_types:
        print(f"\n{'='*60}")
        print(f"Testing {robot_type}")
        print(f"{'='*60}")

        metadata = normalizer.get_robot_metadata(robot_type)
        dof = metadata['dof']
        action_horizon = 16
        joint_dim = 2 * dof  # 12 for 6-DoF dual-arm

        # Create synthetic test data within the q01-q99 range
        np.random.seed(42)

        # Get stats to create reasonable test values
        delta_q01 = normalizer.stats[robot_type]["delta_actions"]["q01"]
        delta_q99 = normalizer.stats[robot_type]["delta_actions"]["q99"]

        # Generate deltas within 80% of the q01-q99 range (to avoid edge effects)
        delta_range = (delta_q99 - delta_q01) * 0.4
        delta_center = (delta_q99 + delta_q01) / 2
        delta_joints = (np.random.rand(action_horizon, joint_dim).astype(np.float32) - 0.5) * delta_range + delta_center

        # Gripper absolute values: 0-1 range
        future_grippers = np.random.rand(action_horizon, 2).astype(np.float32)

        print(f"\nOriginal data:")
        print(f"  Joint deltas shape: {delta_joints.shape}, range: [{delta_joints.min():.4f}, {delta_joints.max():.4f}]")
        print(f"  Future grippers shape: {future_grippers.shape}, range: [{future_grippers.min():.4f}, {future_grippers.max():.4f}]")

        # === TRAINING SIDE: Normalize and concatenate ===
        # (as done in robotwin_dataset.py)
        normalized_joint_deltas = normalizer.normalize_delta_actions(delta_joints, robot_type)
        normalized_future_grippers = normalizer.normalize_grippers(future_grippers, robot_type)

        # Concatenate into combined format
        normalized_combined = np.concatenate([normalized_joint_deltas, normalized_future_grippers], axis=1)

        print(f"\nNormalized combined shape: {normalized_combined.shape}")
        print(f"  Joint deltas ([:, :12]): [{normalized_combined[:, :joint_dim].min():.4f}, {normalized_combined[:, :joint_dim].max():.4f}]")
        print(f"  Grippers ([:, 12:]): [{normalized_combined[:, joint_dim:].min():.4f}, {normalized_combined[:, joint_dim:].max():.4f}]")

        # Values should be mostly in [-1, 1] since we generated data within q01-q99
        assert normalized_combined.min() >= -1.5 and normalized_combined.max() <= 1.5, \
            f"Combined normalized values unexpectedly outside [-1.5, 1.5]: [{normalized_combined.min():.4f}, {normalized_combined.max():.4f}]"

        # === EVALUATION SIDE: Split and denormalize ===
        # (as done in qwen3_vla_policy.py)
        decoded_normalized_joints = normalized_combined[:, :joint_dim]
        decoded_normalized_grippers = normalized_combined[:, joint_dim:]

        # Denormalize separately
        decoded_joint_deltas = normalizer.denormalize_delta_actions(decoded_normalized_joints, robot_type)
        decoded_future_grippers = normalizer.denormalize_grippers(decoded_normalized_grippers, robot_type)

        print(f"\nDecoded data:")
        print(f"  Joint deltas: [{decoded_joint_deltas.min():.4f}, {decoded_joint_deltas.max():.4f}]")
        print(f"  Future grippers: [{decoded_future_grippers.min():.4f}, {decoded_future_grippers.max():.4f}]")

        # === VERIFY ROUNDTRIP ===
        joint_error = np.abs(delta_joints - decoded_joint_deltas).mean()
        gripper_error = np.abs(future_grippers - decoded_future_grippers).mean()

        print(f"\nRoundtrip errors:")
        print(f"  Joint deltas MAE: {joint_error:.6f}")
        print(f"  Future grippers MAE: {gripper_error:.6f}")

        # Roundtrip should be exact (no clamping)
        assert joint_error < 1e-5, f"Joint roundtrip error too large: {joint_error}"
        assert gripper_error < 1e-5, f"Gripper roundtrip error too large: {gripper_error}"

        print(f"\n✓ Combined action roundtrip passed for {robot_type}")

    print("\n" + "=" * 60)
    print("✓ Combined action roundtrip test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_normalization_roundtrip()
    test_combined_action_roundtrip()
